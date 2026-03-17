"""
MindGuard Database Layer
========================
SQLite-based storage for:
- Organization management (multi-tenant)
- API key management
- Aggregate trend history (NEVER individual texts)
- Usage tracking & billing
- Webhook registrations

Privacy Guarantee:
    This database NEVER stores individual text inputs or per-person results.
    Only aggregate statistics (health scores, severity distributions, dimension averages)
    are persisted for trend analysis.
"""

import os
import sqlite3
import hashlib
import secrets
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager


class MindGuardDB:
    """
    Multi-tenant database for MindGuard Community Edition.
    Stores ONLY aggregate data — never individual texts or results.
    """

    def __init__(self, db_path: str = "data/mindguard.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _get_conn(self):
        """Thread-safe connection context manager."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self):
        """Create all tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                -- Organizations (multi-tenant)
                CREATE TABLE IF NOT EXISTS organizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    tier TEXT DEFAULT 'free',
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    settings TEXT DEFAULT '{}'
                );

                -- API Keys (linked to organizations)
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT UNIQUE NOT NULL,
                    key_prefix TEXT NOT NULL,
                    org_id TEXT NOT NULL,
                    name TEXT DEFAULT 'Default',
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now')),
                    last_used_at TEXT,
                    FOREIGN KEY (org_id) REFERENCES organizations(org_id)
                );

                -- Aggregate Trend Snapshots (PRIVACY-SAFE: no individual data)
                CREATE TABLE IF NOT EXISTS trend_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id TEXT NOT NULL,
                    snapshot_date TEXT NOT NULL,
                    total_analyzed INTEGER DEFAULT 0,
                    health_score INTEGER DEFAULT 100,
                    critical_flags INTEGER DEFAULT 0,
                    severity_distribution TEXT DEFAULT '{}',
                    avg_dimensions TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (org_id) REFERENCES organizations(org_id)
                );

                -- Usage Tracking (for billing & rate limiting)
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    count INTEGER DEFAULT 1,
                    date TEXT DEFAULT (date('now')),
                    FOREIGN KEY (org_id) REFERENCES organizations(org_id)
                );

                -- Webhook Registrations
                CREATE TABLE IF NOT EXISTS webhooks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    events TEXT NOT NULL,
                    secret TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (org_id) REFERENCES organizations(org_id)
                );

                -- Alert Configuration per Organization
                CREATE TABLE IF NOT EXISTS alert_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    notification_emails TEXT DEFAULT '[]',
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (org_id) REFERENCES organizations(org_id)
                );

                -- Create indexes
                CREATE INDEX IF NOT EXISTS idx_trend_org_date 
                    ON trend_snapshots(org_id, snapshot_date);
                CREATE INDEX IF NOT EXISTS idx_usage_org_date 
                    ON usage_log(org_id, date);
                CREATE INDEX IF NOT EXISTS idx_apikey_hash 
                    ON api_keys(key_hash);
            """)
            
            # Seed default demo organization
            self._seed_demo_org(conn)

    def _seed_demo_org(self, conn):
        """Create a demo organization for testing."""
        existing = conn.execute(
            "SELECT 1 FROM organizations WHERE org_id = 'demo'",
        ).fetchone()
        
        if not existing:
            conn.execute("""
                INSERT INTO organizations (org_id, name, email, tier)
                VALUES ('demo', 'Demo Organization', 'demo@mindguard.ai', 'starter')
            """)
            # Create demo API key: "mg_demo_organization_key"
            demo_key = "mg_demo_organization_key"
            key_hash = hashlib.sha256(demo_key.encode()).hexdigest()
            conn.execute("""
                INSERT INTO api_keys (key_hash, key_prefix, org_id, name)
                VALUES (?, 'mg_demo_', 'demo', 'Demo Key')
            """, (key_hash,))

    # =========================================
    # Organization Management
    # =========================================

    def create_organization(
        self, name: str, email: str, tier: str = "free"
    ) -> Tuple[str, str]:
        """
        Register a new organization. Returns (org_id, api_key).
        """
        org_id = f"org_{secrets.token_hex(8)}"
        api_key = f"mg_{secrets.token_hex(24)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_prefix = api_key[:8]

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO organizations (org_id, name, email, tier)
                VALUES (?, ?, ?, ?)
            """, (org_id, name, email, tier))
            
            conn.execute("""
                INSERT INTO api_keys (key_hash, key_prefix, org_id, name)
                VALUES (?, ?, ?, 'Primary')
            """, (key_hash, key_prefix, org_id))

        return org_id, api_key

    def get_organization(self, org_id: str) -> Optional[Dict]:
        """Get organization details."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM organizations WHERE org_id = ? AND is_active = 1",
                (org_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_organizations(self) -> List[Dict]:
        """List all active organizations."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM organizations WHERE is_active = 1 ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def update_organization(self, org_id: str, **kwargs) -> bool:
        """Update organization fields."""
        allowed = {'name', 'email', 'tier', 'is_active', 'settings'}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [org_id]
        
        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE organizations SET {set_clause}, updated_at = datetime('now') WHERE org_id = ?",
                values
            )
        return True

    # =========================================
    # API Key Management
    # =========================================

    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """
        Validate an API key. Returns org info if valid, None if invalid.
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT ak.org_id, ak.name as key_name, o.name as org_name, 
                       o.tier, o.is_active as org_active, ak.is_active as key_active
                FROM api_keys ak
                JOIN organizations o ON ak.org_id = o.org_id
                WHERE ak.key_hash = ?
            """, (key_hash,)).fetchone()
            
            if row and row['org_active'] and row['key_active']:
                # Update last_used timestamp
                conn.execute(
                    "UPDATE api_keys SET last_used_at = datetime('now') WHERE key_hash = ?",
                    (key_hash,)
                )
                return dict(row)
        
        return None

    def create_api_key(self, org_id: str, name: str = "API Key") -> str:
        """Generate a new API key for an organization."""
        api_key = f"mg_{secrets.token_hex(24)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_prefix = api_key[:8]
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO api_keys (key_hash, key_prefix, org_id, name)
                VALUES (?, ?, ?, ?)
            """, (key_hash, key_prefix, org_id, name))
        
        return api_key

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        with self._get_conn() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE key_hash = ?",
                (key_hash,)
            )
            return cursor.rowcount > 0

    # =========================================
    # Trend Snapshots (Privacy-Safe Aggregates)
    # =========================================

    def save_trend_snapshot(
        self,
        org_id: str,
        total_analyzed: int,
        health_score: int,
        critical_flags: int,
        severity_distribution: Dict,
        avg_dimensions: Dict,
        snapshot_date: Optional[str] = None,
    ):
        """Save an aggregate trend snapshot. NO individual data stored."""
        if snapshot_date is None:
            snapshot_date = datetime.now().strftime("%Y-%m-%d")

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO trend_snapshots 
                (org_id, snapshot_date, total_analyzed, health_score, 
                 critical_flags, severity_distribution, avg_dimensions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                org_id, snapshot_date, total_analyzed, health_score,
                critical_flags,
                json.dumps(severity_distribution),
                json.dumps(avg_dimensions),
            ))

    def get_trend_history(
        self, org_id: str, days: int = 30
    ) -> List[Dict]:
        """Get trend history for the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM trend_snapshots
                WHERE org_id = ? AND snapshot_date >= ?
                ORDER BY snapshot_date ASC
            """, (org_id, cutoff)).fetchall()
            
            results = []
            for r in rows:
                d = dict(r)
                d['severity_distribution'] = json.loads(d.get('severity_distribution', '{}'))
                d['avg_dimensions'] = json.loads(d.get('avg_dimensions', '{}'))
                results.append(d)
            return results

    def get_latest_snapshot(self, org_id: str) -> Optional[Dict]:
        """Get the most recent trend snapshot."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM trend_snapshots
                WHERE org_id = ?
                ORDER BY snapshot_date DESC LIMIT 1
            """, (org_id,)).fetchone()
            
            if row:
                d = dict(row)
                d['severity_distribution'] = json.loads(d.get('severity_distribution', '{}'))
                d['avg_dimensions'] = json.loads(d.get('avg_dimensions', '{}'))
                return d
        return None

    # =========================================
    # Usage Tracking
    # =========================================

    def log_usage(self, org_id: str, endpoint: str, count: int = 1):
        """Log API usage for rate limiting and billing."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        with self._get_conn() as conn:
            existing = conn.execute("""
                SELECT id, count FROM usage_log
                WHERE org_id = ? AND endpoint = ? AND date = ?
            """, (org_id, endpoint, today)).fetchone()
            
            if existing:
                conn.execute(
                    "UPDATE usage_log SET count = count + ? WHERE id = ?",
                    (count, existing['id'])
                )
            else:
                conn.execute(
                    "INSERT INTO usage_log (org_id, endpoint, count, date) VALUES (?, ?, ?, ?)",
                    (org_id, endpoint, count, today)
                )

    def get_usage_stats(self, org_id: str, days: int = 30) -> Dict:
        """Get usage statistics for billing."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT endpoint, SUM(count) as total, date
                FROM usage_log
                WHERE org_id = ? AND date >= ?
                GROUP BY endpoint, date
                ORDER BY date ASC
            """, (org_id, cutoff)).fetchall()
            
            total = sum(r['total'] for r in rows)
            by_endpoint = {}
            by_date = {}
            
            for r in rows:
                ep = r['endpoint']
                if ep not in by_endpoint:
                    by_endpoint[ep] = 0
                by_endpoint[ep] += r['total']
                
                dt = r['date']
                if dt not in by_date:
                    by_date[dt] = 0
                by_date[dt] += r['total']
            
            return {
                'total_requests': total,
                'by_endpoint': by_endpoint,
                'by_date': by_date,
                'period_days': days,
            }

    def check_rate_limit(self, org_id: str, tier_limits: Dict) -> Dict:
        """Check if org is within rate limits based on their tier."""
        today = datetime.now().strftime("%Y-%m-%d")
        month_start = datetime.now().strftime("%Y-%m-01")
        
        with self._get_conn() as conn:
            # Monthly total
            row = conn.execute("""
                SELECT COALESCE(SUM(count), 0) as monthly_total
                FROM usage_log WHERE org_id = ? AND date >= ?
            """, (org_id, month_start)).fetchone()
            
            monthly = row['monthly_total'] if row else 0
            max_monthly = tier_limits.get('max_analyses_per_month', 500)
            
            return {
                'allowed': max_monthly == -1 or monthly < max_monthly,
                'monthly_used': monthly,
                'monthly_limit': max_monthly,
                'remaining': max(0, max_monthly - monthly) if max_monthly != -1 else -1,
            }

    # =========================================
    # Webhook Management
    # =========================================

    def register_webhook(
        self, org_id: str, url: str, events: List[str]
    ) -> int:
        """Register a webhook for an organization."""
        secret = secrets.token_hex(32)
        with self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO webhooks (org_id, url, events, secret)
                VALUES (?, ?, ?, ?)
            """, (org_id, url, json.dumps(events), secret))
            return cursor.lastrowid

    def get_webhooks(self, org_id: str, event: Optional[str] = None) -> List[Dict]:
        """Get active webhooks, optionally filtered by event type."""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM webhooks
                WHERE org_id = ? AND is_active = 1
            """, (org_id,)).fetchall()
            
            results = []
            for r in rows:
                d = dict(r)
                d['events'] = json.loads(d.get('events', '[]'))
                if event is None or event in d['events']:
                    results.append(d)
            return results

    # =========================================
    # Admin Analytics
    # =========================================

    def get_platform_stats(self) -> Dict:
        """Platform-wide statistics for admin dashboard."""
        with self._get_conn() as conn:
            orgs = conn.execute(
                "SELECT COUNT(*) as c FROM organizations WHERE is_active = 1"
            ).fetchone()['c']
            
            keys = conn.execute(
                "SELECT COUNT(*) as c FROM api_keys WHERE is_active = 1"
            ).fetchone()['c']
            
            month_start = datetime.now().strftime("%Y-%m-01")
            usage = conn.execute("""
                SELECT COALESCE(SUM(count), 0) as c FROM usage_log WHERE date >= ?
            """, (month_start,)).fetchone()['c']
            
            snapshots = conn.execute(
                "SELECT COUNT(*) as c FROM trend_snapshots"
            ).fetchone()['c']
            
            tier_dist = {}
            for row in conn.execute(
                "SELECT tier, COUNT(*) as c FROM organizations WHERE is_active = 1 GROUP BY tier"
            ).fetchall():
                tier_dist[row['tier']] = row['c']
            
            return {
                'total_organizations': orgs,
                'total_api_keys': keys,
                'monthly_analyses': usage,
                'total_snapshots': snapshots,
                'tier_distribution': tier_dist,
            }
