"""
Automated System Audit - Tests the multi-agent system with real scenarios.

This script simulates user interactions and verifies:
- Policy questions are answered correctly
- Order eligibility is checked correctly
- Already returned orders are handled properly
- Multi-item orders are processed correctly
- Edge cases are handled
"""
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse
from src.agents import CoordinatorAgent, PolicyExpertAgent, TransactionAgent, AgentRequest
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SystemAuditor:
    """Automated auditor for the multi-agent system."""

    def __init__(self):
        self.langfuse = Langfuse()
        self.policy_expert = PolicyExpertAgent(tracer=self.langfuse)
        self.transaction_agent = TransactionAgent(tracer=self.langfuse)
        self.coordinator = CoordinatorAgent(
            tracer=self.langfuse,
            specialized_agents={
                "policy_expert": self.policy_expert,
                "transaction_agent": self.transaction_agent
            }
        )
        self.test_results = []

    async def test_query(self, test_name: str, user_message: str, expected_behavior: dict) -> dict:
        """
        Test a single user query.

        Args:
            test_name: Name of the test
            user_message: User's message
            expected_behavior: Dict with expected response characteristics

        Returns:
            Test result dict
        """
        print(f"\n{'='*70}")
        print(f"🧪 TEST: {test_name}")
        print(f"{'='*70}")
        print(f"📝 Query: {user_message}")

        request = AgentRequest(
            agent="coordinator",
            task="handle_user_query",
            context={"user_message": user_message},
            metadata={"test_name": test_name}
        )

        start_time = datetime.now()
        response = await self.coordinator.handle_request(request)
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        result = {
            "test_name": test_name,
            "user_message": user_message,
            "status": response.status,
            "latency_ms": latency_ms,
            "passed": True,
            "issues": []
        }

        if response.status != "success":
            result["passed"] = False
            result["issues"].append(f"Request failed: {response.error}")
            print(f"❌ FAILED: {response.error}")
            return result

        # Extract response details
        response_data = response.result
        response_template = response_data.get('response')

        print(f"\n📊 Response Type: {response_template.response_type}")
        print(f"⏱️  Latency: {latency_ms:.0f}ms")
        print(f"🤖 Agents Called: {', '.join(response_data.get('agents_called', []))}")
        print(f"\n💬 Message:\n{response_template.message}")

        if response_template.key_details:
            print(f"\n📌 Key Details:")
            for detail in response_template.key_details:
                print(f"  • {detail}")

        # Verify expected behavior
        if "expected_response_type" in expected_behavior:
            expected_type = expected_behavior["expected_response_type"]
            if response_template.response_type != expected_type:
                result["passed"] = False
                result["issues"].append(
                    f"Expected response_type '{expected_type}', got '{response_template.response_type}'"
                )

        if "should_include_keywords" in expected_behavior:
            message_lower = response_template.message.lower()
            for keyword in expected_behavior["should_include_keywords"]:
                if keyword.lower() not in message_lower:
                    result["passed"] = False
                    result["issues"].append(f"Missing expected keyword: '{keyword}'")

        if "should_not_include_keywords" in expected_behavior:
            message_lower = response_template.message.lower()
            for keyword in expected_behavior["should_not_include_keywords"]:
                if keyword.lower() in message_lower:
                    result["passed"] = False
                    result["issues"].append(f"Should not include keyword: '{keyword}'")

        if "expected_eligible" in expected_behavior:
            eligibility_info = response_data.get('eligibility_info')
            if eligibility_info:
                if eligibility_info.eligible != expected_behavior["expected_eligible"]:
                    result["passed"] = False
                    result["issues"].append(
                        f"Expected eligible={expected_behavior['expected_eligible']}, "
                        f"got {eligibility_info.eligible}"
                    )

        # Print result
        if result["passed"]:
            print(f"\n✅ TEST PASSED")
        else:
            print(f"\n❌ TEST FAILED")
            for issue in result["issues"]:
                print(f"   ⚠️  {issue}")

        self.test_results.append(result)
        return result

    async def run_audit(self):
        """Run complete audit with all test cases."""
        print("\n" + "="*70)
        print("🔍 MULTI-AGENT SYSTEM AUDIT")
        print("="*70)

        # Test 1: General policy question (14 days)
        await self.test_query(
            test_name="Policy: 14-day return window",
            user_message="¿Cuántos días tengo para devolver un producto?",
            expected_behavior={
                "expected_response_type": "policy_info",
                "should_include_keywords": ["14 días", "desistimiento"]
            }
        )

        # Test 2: Product condition requirements
        await self.test_query(
            test_name="Policy: Product condition",
            user_message="¿Puedo devolver zapatos si los he usado en exteriores?",
            expected_behavior={
                "expected_response_type": "policy_info",
                "should_include_keywords": ["interiores", "perfecto estado"]
            }
        )

        # Test 3: Already returned order (ORD-159753)
        await self.test_query(
            test_name="Order: Already returned (ORD-159753)",
            user_message="Quiero devolver mi pedido ORD-159753",
            expected_behavior={
                "expected_response_type": "refund_already_processed",
                "should_include_keywords": ["ya", "procesado", "reembolso"],
                "should_not_include_keywords": ["confirmar", "proceder"]
            }
        )

        # Test 4: Eligible order with 2 items (ORD-295481)
        await self.test_query(
            test_name="Order: Eligible with 2 items (ORD-295481)",
            user_message="Quiero devolver mi pedido ORD-295481",
            expected_behavior={
                "expected_response_type": "refund_eligible",
                "expected_eligible": True,
                "should_include_keywords": ["elegible", "devolución"]
            }
        )

        # Test 5: Not eligible - old and pending (ORD-99887)
        await self.test_query(
            test_name="Order: Not eligible - date & status (ORD-99887)",
            user_message="Quiero devolver mi pedido ORD-99887",
            expected_behavior={
                "expected_response_type": "refund_not_eligible",
                "expected_eligible": False,
                "should_include_keywords": ["no cumple", "14 días"]
            }
        )

        # Test 6: Packaging requirements
        await self.test_query(
            test_name="Policy: Packaging requirements",
            user_message="¿Necesito devolver los zapatos en la caja original?",
            expected_behavior={
                "expected_response_type": "policy_info",
                "should_include_keywords": ["caja original", "embalaje"]
            }
        )

        # Test 7: Warranty/defects
        await self.test_query(
            test_name="Policy: Warranty for defects",
            user_message="¿Qué garantía tienen los productos?",
            expected_behavior={
                "expected_response_type": "policy_info",
                "should_include_keywords": ["3 años", "garantía", "defectos"]
            }
        )

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate final audit report."""
        print("\n" + "="*70)
        print("📋 AUDIT REPORT")
        print("="*70)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results if t["passed"])
        failed_tests = total_tests - passed_tests

        avg_latency = sum(t["latency_ms"] for t in self.test_results) / total_tests

        print(f"\n✅ Tests Passed: {passed_tests}/{total_tests}")
        print(f"❌ Tests Failed: {failed_tests}/{total_tests}")
        print(f"⏱️  Average Latency: {avg_latency:.0f}ms")

        if failed_tests > 0:
            print(f"\n{'='*70}")
            print("🚨 FAILED TESTS DETAILS")
            print("="*70)
            for result in self.test_results:
                if not result["passed"]:
                    print(f"\n❌ {result['test_name']}")
                    for issue in result["issues"]:
                        print(f"   • {issue}")

        # Save detailed report
        report_file = "audit_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed report saved to: {report_file}")

        print("\n" + "="*70)
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED - SYSTEM IS HEALTHY")
        else:
            print("⚠️  ISSUES DETECTED - REVIEW REQUIRED")
        print("="*70)


async def main():
    """Run the audit."""
    auditor = SystemAuditor()
    await auditor.run_audit()


if __name__ == "__main__":
    asyncio.run(main())
