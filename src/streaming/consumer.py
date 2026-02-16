# src/streaming/consumer.py
"""
Kafka Consumer for Real-Time Transaction Processing.
Processes bank transaction events, updates running feature aggregations,
and triggers re-scoring when significant changes are detected.

Architecture:
  Kafka Topic → Consumer → Feature Update → Re-Score → Alert
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Thresholds for triggering re-scoring
RESCORE_THRESHOLDS = {
    'large_expense_ratio': 0.3,      # Single expense > 30% of monthly income
    'balance_drop_pct': 0.4,         # Balance drops > 40% in one day
    'consecutive_failures': 2,        # 2+ payment failures in a row
    'salary_delay_days': 5,           # Salary > 5 days late
    'health_expense_spike': 3.0,      # Healthcare > 3x baseline
}


class TransactionConsumer:
    """Kafka consumer for real-time bank transaction events.

    Processes transaction events from a Kafka topic, maintains running
    aggregations per customer, and triggers re-scoring when patterns
    change significantly.

    Usage:
        consumer = TransactionConsumer(
            bootstrap_servers='localhost:9092',
            topic='bank_transactions'
        )
        consumer.start()
    """

    def __init__(self, bootstrap_servers='localhost:9092',
                 topic='bank_transactions', group_id='predelinq-consumer'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer = None
        self.running_aggregations: Dict[str, dict] = {}

    def connect(self):
        """Connect to Kafka broker."""
        try:
            from kafka import KafkaConsumer
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                max_poll_records=100,
            )
            logger.info(f"Connected to Kafka: {self.bootstrap_servers}, topic: {self.topic}")
        except ImportError:
            logger.warning("kafka-python not installed. Using stub mode.")
            self.consumer = None
        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")
            self.consumer = None

    def process_transaction(self, event: dict) -> Optional[dict]:
        """Process a single transaction event.

        Args:
            event: Transaction data with fields:
                - customer_id: str
                - type: 'credit' | 'debit' | 'payment' | 'salary'
                - amount: float
                - category: str (e.g., 'healthcare', 'entertainment')
                - timestamp: ISO datetime string

        Returns:
            Alert dict if re-scoring is triggered, else None.
        """
        cid = event.get('customer_id')
        if not cid:
            return None

        # Initialize customer aggregation
        if cid not in self.running_aggregations:
            self.running_aggregations[cid] = {
                'total_credits': 0,
                'total_debits': 0,
                'current_balance_estimate': 0,
                'transaction_count': 0,
                'payment_failures': 0,
                'consecutive_failures': 0,
                'last_salary_date': None,
                'healthcare_spend_30d': 0,
                'baseline_monthly_income': event.get('monthly_income', 50000),
            }

        agg = self.running_aggregations[cid]
        tx_type = event.get('type', 'debit')
        amount = event.get('amount', 0)
        category = event.get('category', 'other')

        # Update aggregations
        agg['transaction_count'] += 1
        if tx_type in ('credit', 'salary'):
            agg['total_credits'] += amount
            agg['current_balance_estimate'] += amount
            if tx_type == 'salary':
                agg['last_salary_date'] = event.get('timestamp')
                agg['consecutive_failures'] = 0  # Reset on salary received
        elif tx_type == 'debit':
            agg['total_debits'] += amount
            agg['current_balance_estimate'] -= amount
        elif tx_type == 'payment_failure':
            agg['payment_failures'] += 1
            agg['consecutive_failures'] += 1

        # Category-specific tracking
        if category == 'healthcare':
            agg['healthcare_spend_30d'] += amount

        # Check re-scoring thresholds
        alert = self._check_thresholds(cid, event, agg)
        return alert

    def _check_thresholds(self, customer_id: str, event: dict, agg: dict) -> Optional[dict]:
        """Check if re-scoring should be triggered."""
        amount = event.get('amount', 0)
        monthly_income = agg['baseline_monthly_income']
        reasons = []

        # Large single expense
        if amount / max(monthly_income, 1) > RESCORE_THRESHOLDS['large_expense_ratio']:
            reasons.append(f"Large expense: ₹{amount:,.0f} ({amount/monthly_income*100:.0f}% of income)")

        # Consecutive payment failures
        if agg['consecutive_failures'] >= RESCORE_THRESHOLDS['consecutive_failures']:
            reasons.append(f"Consecutive payment failures: {agg['consecutive_failures']}")

        # Healthcare spike
        baseline = monthly_income * 0.05  # 5% baseline healthcare
        if agg['healthcare_spend_30d'] / max(baseline, 1) > RESCORE_THRESHOLDS['health_expense_spike']:
            spike = agg['healthcare_spend_30d'] / max(baseline, 1)
            reasons.append(f"Healthcare spike: {spike:.1f}x baseline")

        if reasons:
            alert = {
                'customer_id': customer_id,
                'trigger_time': datetime.utcnow().isoformat(),
                'reasons': reasons,
                'action': 'RESCORE',
                'priority': 'HIGH' if len(reasons) > 1 else 'MEDIUM',
            }
            logger.warning(f"Re-scoring triggered for {customer_id}: {reasons}")
            return alert

        return None

    def start(self, max_messages=None):
        """Start consuming messages.

        Args:
            max_messages: Stop after this many messages (None for infinite).
        """
        self.connect()
        if self.consumer is None:
            logger.warning("No Kafka consumer available. Running in dry mode.")
            return

        count = 0
        logger.info("Starting transaction consumer...")
        try:
            for message in self.consumer:
                event = message.value
                alert = self.process_transaction(event)
                if alert:
                    self._handle_alert(alert)

                count += 1
                if max_messages and count >= max_messages:
                    break
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user.")
        finally:
            if self.consumer:
                self.consumer.close()
            logger.info(f"Consumer stopped. Processed {count} messages.")

    def _handle_alert(self, alert: dict):
        """Handle a re-scoring alert (send to scoring service)."""
        logger.info(f"ALERT: {alert['customer_id']} - {alert['reasons']}")
        # In production: POST to /risk-score API endpoint
        # requests.post('http://localhost:8000/risk-score', json=customer_data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    consumer = TransactionConsumer()
    consumer.start()
