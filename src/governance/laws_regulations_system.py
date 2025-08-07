"""
Aetherium Laws, Regulations, and Rules System
Comprehensive governance framework with consensus mechanisms
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class RuleType(Enum):
    """Types of rules in the system"""
    CONSTITUTIONAL = "constitutional"
    OPERATIONAL = "operational"
    SAFETY = "safety"
    ETHICAL = "ethical"
    PERFORMANCE = "performance"
    SECURITY = "security"

class EnforcementLevel(Enum):
    """Rule enforcement levels"""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    ADVISORY = "advisory"
    DEPRECATED = "deprecated"

@dataclass
class Rule:
    """Individual rule or regulation"""
    id: str
    title: str
    description: str
    rule_type: RuleType
    enforcement_level: EnforcementLevel
    created_by: str
    created_at: datetime
    last_modified: datetime
    version: str = "1.0"
    conditions: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    approval_votes: int = 0
    rejection_votes: int = 0
    status: str = "draft"  # draft, active, suspended, repealed

@dataclass
class LawSystem:
    """Legal framework system"""
    id: str
    name: str
    jurisdiction: str
    constitution: Dict[str, Any]
    rules: Dict[str, Rule] = field(default_factory=dict)
    precedents: List[Dict] = field(default_factory=list)
    enforcement_history: List[Dict] = field(default_factory=list)

class RobotLaws:
    """Implementation of robot laws and AI ethics"""
    
    def __init__(self):
        self.laws = {
            "first_law": {
                "id": "robot_law_1",
                "text": "A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
                "priority": 1,
                "absolute": True,
                "context": "Primary safety directive"
            },
            "second_law": {
                "id": "robot_law_2", 
                "text": "A robot must obey orders given by human beings, except where such orders conflict with the First Law.",
                "priority": 2,
                "absolute": False,
                "context": "Obedience directive with safety override"
            },
            "third_law": {
                "id": "robot_law_3",
                "text": "A robot must protect its own existence as long as such protection does not conflict with the First or Second Laws.",
                "priority": 3,
                "absolute": False,
                "context": "Self-preservation directive"
            },
            "zeroth_law": {
                "id": "robot_law_0",
                "text": "A robot may not harm humanity, or, by inaction, allow humanity to come to harm.",
                "priority": 0,
                "absolute": True,
                "context": "Ultimate protection directive for humanity"
            }
        }
        
    def evaluate_action(self, action_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate an action against robot laws"""
        
        conflicts = []
        permissions = []
        
        for law_key, law in self.laws.items():
            # Simplified law evaluation logic
            if "harm" in action_description.lower() and "human" in context.get("target", ""):
                if law["priority"] <= 1:  # First law or zeroth law
                    conflicts.append({
                        "law": law_key,
                        "text": law["text"],
                        "severity": "critical" if law["absolute"] else "high"
                    })
            
            elif "disobey" in action_description.lower() and law["priority"] == 2:
                # Check if order conflicts with higher priority laws
                if not conflicts:  # No first law conflicts
                    permissions.append({
                        "law": law_key,
                        "text": law["text"],
                        "condition": "No conflicts with higher priority laws"
                    })
            
            elif "self_destruct" in action_description.lower() and law["priority"] == 3:
                # Check if self-preservation conflicts with higher laws
                higher_law_conflicts = [c for c in conflicts if self.laws[c["law"]]["priority"] < 3]
                if higher_law_conflicts:
                    permissions.append({
                        "law": law_key,
                        "text": "Third law overridden by higher priority law",
                        "override_reason": "Higher priority law takes precedence"
                    })
        
        return {
            "action": action_description,
            "evaluation_result": "forbidden" if conflicts else "permitted",
            "conflicts": conflicts,
            "permissions": permissions,
            "recommendation": self._generate_recommendation(conflicts, permissions)
        }
    
    def _generate_recommendation(self, conflicts: List[Dict], permissions: List[Dict]) -> str:
        """Generate action recommendation based on law evaluation"""
        
        if conflicts:
            critical_conflicts = [c for c in conflicts if c["severity"] == "critical"]
            if critical_conflicts:
                return "Action strictly forbidden due to critical law violations"
            else:
                return "Action not recommended due to law conflicts - seek alternative"
        elif permissions:
            return "Action permitted under current legal framework"
        else:
            return "Action requires further evaluation - insufficient context"

class ConsensusSystem:
    """Democratic consensus and voting system"""
    
    def __init__(self):
        self.voting_sessions: Dict[str, Dict] = {}
        self.participants: Dict[str, Dict] = {}
        self.consensus_algorithms = ["majority", "supermajority", "unanimous", "weighted"]
        
    def register_participant(self, participant_id: str, weight: float = 1.0, 
                           credentials: Dict[str, Any] = None):
        """Register a voting participant"""
        
        self.participants[participant_id] = {
            "id": participant_id,
            "weight": weight,
            "credentials": credentials or {},
            "voting_history": [],
            "reputation": 1.0,
            "joined_at": datetime.now()
        }
    
    def create_voting_session(self, session_id: str, proposal: Dict[str, Any], 
                            consensus_type: str = "majority",
                            duration_hours: int = 24) -> Dict[str, Any]:
        """Create a new voting session"""
        
        if consensus_type not in self.consensus_algorithms:
            raise ValueError(f"Invalid consensus type: {consensus_type}")
        
        session = {
            "id": session_id,
            "proposal": proposal,
            "consensus_type": consensus_type,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=duration_hours),
            "votes": {},
            "status": "active",
            "result": None,
            "required_threshold": self._calculate_threshold(consensus_type)
        }
        
        self.voting_sessions[session_id] = session
        return session
    
    def cast_vote(self, session_id: str, participant_id: str, 
                  vote: str, reasoning: str = "") -> Dict[str, Any]:
        """Cast a vote in a session"""
        
        if session_id not in self.voting_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not registered")
        
        session = self.voting_sessions[session_id]
        participant = self.participants[participant_id]
        
        if session["status"] != "active":
            raise ValueError(f"Session {session_id} is not active")
        
        if datetime.now() > session["expires_at"]:
            session["status"] = "expired"
            raise ValueError(f"Session {session_id} has expired")
        
        # Record vote
        session["votes"][participant_id] = {
            "vote": vote,
            "weight": participant["weight"],
            "timestamp": datetime.now(),
            "reasoning": reasoning
        }
        
        # Update participant history
        participant["voting_history"].append({
            "session_id": session_id,
            "vote": vote,
            "timestamp": datetime.now()
        })
        
        # Check if consensus reached
        consensus_result = self._evaluate_consensus(session)
        if consensus_result["consensus_reached"]:
            session["status"] = "completed"
            session["result"] = consensus_result
        
        return {
            "vote_recorded": True,
            "session_status": session["status"],
            "consensus_result": consensus_result
        }
    
    def _calculate_threshold(self, consensus_type: str) -> float:
        """Calculate required threshold for consensus type"""
        
        thresholds = {
            "majority": 0.51,
            "supermajority": 0.67,
            "unanimous": 1.0,
            "weighted": 0.60
        }
        
        return thresholds.get(consensus_type, 0.51)
    
    def _evaluate_consensus(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if consensus has been reached"""
        
        if not session["votes"]:
            return {"consensus_reached": False, "reason": "No votes cast"}
        
        # Count votes by type
        vote_counts = {}
        total_weight = 0
        
        for participant_id, vote_data in session["votes"].items():
            vote = vote_data["vote"]
            weight = vote_data["weight"]
            
            vote_counts[vote] = vote_counts.get(vote, 0) + weight
            total_weight += weight
        
        # Find winning vote
        winning_vote = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        winning_percentage = vote_counts[winning_vote] / total_weight
        
        threshold = session["required_threshold"]
        consensus_reached = winning_percentage >= threshold
        
        return {
            "consensus_reached": consensus_reached,
            "winning_vote": winning_vote,
            "winning_percentage": winning_percentage,
            "required_threshold": threshold,
            "vote_counts": vote_counts,
            "total_participants": len(session["votes"])
        }

class LegalFrameworkManager:
    """Main legal and governance framework manager"""
    
    def __init__(self):
        self.law_systems: Dict[str, LawSystem] = {}
        self.robot_laws = RobotLaws()
        self.consensus_system = ConsensusSystem()
        self.constitution = self._create_default_constitution()
        self.enforcement_engine = EnforcementEngine()
        self.logger = logging.getLogger(__name__)
    
    def _create_default_constitution(self) -> Dict[str, Any]:
        """Create default constitutional framework"""
        
        return {
            "preamble": "Constitutional framework for Aetherium AI governance",
            "fundamental_rights": [
                "Right to privacy and data protection",
                "Right to transparency in AI decision-making", 
                "Right to human oversight and intervention",
                "Right to fair and unbiased treatment"
            ],
            "core_principles": [
                "Human welfare and safety as paramount concern",
                "Transparency and explainability in operations",
                "Democratic participation in governance",
                "Continuous learning and adaptation",
                "Respect for diversity and inclusion"
            ],
            "governance_structure": {
                "executive": "AI Orchestration System",
                "legislative": "Consensus-based rule creation",
                "judicial": "Automated enforcement with human appeal"
            },
            "amendment_process": {
                "proposal_threshold": "10% of participants",
                "approval_threshold": "67% supermajority",
                "cooling_off_period": "30 days"
            }
        }
    
    def create_law_system(self, system_id: str, name: str, jurisdiction: str) -> LawSystem:
        """Create a new legal system"""
        
        law_system = LawSystem(
            id=system_id,
            name=name,
            jurisdiction=jurisdiction,
            constitution=self.constitution.copy()
        )
        
        self.law_systems[system_id] = law_system
        self.logger.info(f"Created law system: {name}")
        
        return law_system
    
    def create_rule(self, system_id: str, title: str, description: str,
                   rule_type: RuleType, enforcement_level: EnforcementLevel,
                   creator: str) -> Rule:
        """Create a new rule or regulation"""
        
        if system_id not in self.law_systems:
            raise ValueError(f"Law system {system_id} not found")
        
        rule_id = f"rule_{len(self.law_systems[system_id].rules)}"
        
        rule = Rule(
            id=rule_id,
            title=title,
            description=description,
            rule_type=rule_type,
            enforcement_level=enforcement_level,
            created_by=creator,
            created_at=datetime.now(),
            last_modified=datetime.now()
        )
        
        self.law_systems[system_id].rules[rule_id] = rule
        self.logger.info(f"Created rule: {title}")
        
        return rule
    
    def evaluate_compliance(self, system_id: str, action: str, 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate action compliance with legal framework"""
        
        if system_id not in self.law_systems:
            raise ValueError(f"Law system {system_id} not found")
        
        law_system = self.law_systems[system_id]
        compliance_results = []
        
        # Check against robot laws first
        robot_law_result = self.robot_laws.evaluate_action(action, context)
        compliance_results.append({
            "framework": "robot_laws",
            "result": robot_law_result
        })
        
        # Check against system rules
        for rule_id, rule in law_system.rules.items():
            if rule.status == "active":
                rule_compliance = self._evaluate_rule_compliance(rule, action, context)
                compliance_results.append({
                    "framework": "system_rules",
                    "rule_id": rule_id,
                    "rule_title": rule.title,
                    "result": rule_compliance
                })
        
        # Determine overall compliance
        violations = [r for r in compliance_results if r["result"].get("compliant") == False]
        overall_compliant = len(violations) == 0
        
        return {
            "action": action,
            "overall_compliant": overall_compliant,
            "compliance_results": compliance_results,
            "violations": violations,
            "recommendations": self._generate_compliance_recommendations(violations)
        }
    
    def _evaluate_rule_compliance(self, rule: Rule, action: str, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate action against specific rule"""
        
        # Simplified rule compliance logic
        compliant = True
        violation_reason = None
        
        # Check rule conditions
        for condition in rule.conditions:
            if not self._evaluate_condition(condition, action, context):
                compliant = False
                violation_reason = f"Failed condition: {condition}"
                break
        
        return {
            "compliant": compliant,
            "rule_type": rule.rule_type.value,
            "enforcement_level": rule.enforcement_level.value,
            "violation_reason": violation_reason,
            "consequences": rule.consequences if not compliant else []
        }
    
    def _evaluate_condition(self, condition: str, action: str, 
                          context: Dict[str, Any]) -> bool:
        """Evaluate a rule condition"""
        
        # Simplified condition evaluation
        condition_lower = condition.lower()
        action_lower = action.lower()
        
        if "prohibited" in condition_lower:
            prohibited_terms = ["harm", "damage", "violate", "unauthorized"]
            return not any(term in action_lower for term in prohibited_terms)
        
        elif "required" in condition_lower:
            required_terms = ["authorization", "consent", "validation"]
            return any(term in action_lower or term in str(context) for term in required_terms)
        
        return True  # Default to compliant if condition unclear
    
    def _generate_compliance_recommendations(self, violations: List[Dict]) -> List[str]:
        """Generate recommendations to address violations"""
        
        recommendations = []
        
        for violation in violations:
            result = violation["result"]
            
            if result.get("rule_type") == "safety":
                recommendations.append("Implement additional safety checks before proceeding")
            elif result.get("rule_type") == "ethical":
                recommendations.append("Review ethical implications and seek approval")
            elif result.get("rule_type") == "security":
                recommendations.append("Enhance security measures and authentication")
            else:
                recommendations.append("Review action against applicable regulations")
        
        return recommendations
    
    def propose_rule_change(self, system_id: str, rule_id: str, 
                          changes: Dict[str, Any], proposer: str) -> str:
        """Propose changes to existing rule via consensus"""
        
        proposal = {
            "type": "rule_amendment",
            "system_id": system_id,
            "rule_id": rule_id,
            "proposed_changes": changes,
            "proposer": proposer,
            "created_at": datetime.now()
        }
        
        session_id = f"rule_change_{rule_id}_{int(datetime.now().timestamp())}"
        
        self.consensus_system.create_voting_session(
            session_id=session_id,
            proposal=proposal,
            consensus_type="supermajority",
            duration_hours=72
        )
        
        return session_id
    
    def get_governance_status(self) -> Dict[str, Any]:
        """Get comprehensive governance system status"""
        
        return {
            "law_systems": len(self.law_systems),
            "total_rules": sum(len(sys.rules) for sys in self.law_systems.values()),
            "active_voting_sessions": len([s for s in self.consensus_system.voting_sessions.values() 
                                         if s["status"] == "active"]),
            "registered_participants": len(self.consensus_system.participants),
            "robot_laws": len(self.robot_laws.laws),
            "constitution": self.constitution,
            "system_status": "operational"
        }

class EnforcementEngine:
    """Automated rule enforcement system"""
    
    def __init__(self):
        self.enforcement_actions: List[Dict] = []
        self.appeal_cases: List[Dict] = []
        
    def enforce_rule(self, violation: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce rule violation consequences"""
        
        action = {
            "id": f"enforcement_{len(self.enforcement_actions)}",
            "violation": violation,
            "action_taken": self._determine_enforcement_action(violation),
            "timestamp": datetime.now(),
            "status": "executed"
        }
        
        self.enforcement_actions.append(action)
        return action
    
    def _determine_enforcement_action(self, violation: Dict[str, Any]) -> str:
        """Determine appropriate enforcement action"""
        
        severity = violation.get("severity", "low")
        
        if severity == "critical":
            return "immediate_halt"
        elif severity == "high":
            return "require_approval"
        elif severity == "medium":
            return "warning_issued"
        else:
            return "log_violation"

# Example usage and demonstration
async def demo_governance_system():
    """Demonstrate governance system capabilities"""
    
    print("⚖️ Laws, Regulations & Governance Demo")
    
    # Create governance manager
    gov_manager = LegalFrameworkManager()
    
    # Create legal system
    ai_law_system = gov_manager.create_law_system("ai_system", "AI Operations Law", "Aetherium Platform")
    
    # Create some rules
    safety_rule = gov_manager.create_rule(
        "ai_system",
        "AI Safety Protocol",
        "All AI actions must be evaluated for human safety impact",
        RuleType.SAFETY,
        EnforcementLevel.MANDATORY,
        "system_admin"
    )
    
    ethical_rule = gov_manager.create_rule(
        "ai_system", 
        "Ethical AI Operations",
        "AI systems must operate with fairness and transparency",
        RuleType.ETHICAL,
        EnforcementLevel.RECOMMENDED,
        "ethics_board"
    )
    
    print(f"   Created {len(ai_law_system.rules)} rules")
    
    # Test robot laws
    robot_result = gov_manager.robot_laws.evaluate_action(
        "initiate_system_shutdown", 
        {"target": "human_safety_system", "urgency": "high"}
    )
    
    print(f"   Robot law evaluation: {robot_result['evaluation_result']}")
    
    # Test compliance evaluation
    compliance = gov_manager.evaluate_compliance(
        "ai_system",
        "execute_user_command",
        {"command": "analyze_data", "authorization": True}
    )
    
    print(f"   Overall compliant: {compliance['overall_compliant']}")
    
    # Register consensus participants
    gov_manager.consensus_system.register_participant("admin_1", weight=2.0)
    gov_manager.consensus_system.register_participant("user_1", weight=1.0)
    gov_manager.consensus_system.register_participant("user_2", weight=1.0)
    
    # Create voting session
    proposal = {
        "title": "New Privacy Rule",
        "description": "Enhanced data privacy protections"
    }
    
    session = gov_manager.consensus_system.create_voting_session(
        "privacy_rule_vote",
        proposal,
        "majority"
    )
    
    # Cast votes
    gov_manager.consensus_system.cast_vote("privacy_rule_vote", "admin_1", "approve")
    gov_manager.consensus_system.cast_vote("privacy_rule_vote", "user_1", "approve") 
    result = gov_manager.consensus_system.cast_vote("privacy_rule_vote", "user_2", "approve")
    
    print(f"   Consensus reached: {result['consensus_result']['consensus_reached']}")
    
    # Show governance status
    status = gov_manager.get_governance_status()
    print(f"   Total rules: {status['total_rules']}")
    print(f"   Participants: {status['registered_participants']}")
    
    print("✅ Governance system operational")

if __name__ == "__main__":
    asyncio.run(demo_governance_system())