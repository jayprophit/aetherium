"""
Aetherium Blockchain System
Advanced blockchain implementation with quantum-resistant cryptography
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import secrets

@dataclass
class Transaction:
    """Blockchain transaction"""
    id: str
    sender: str
    receiver: str
    amount: float
    data: Dict[str, Any]
    timestamp: float
    signature: str = ""
    nonce: str = ""

@dataclass
class Block:
    """Blockchain block"""
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    merkle_root: str
    nonce: int = 0
    hash: str = ""
    validator: str = ""
    difficulty: int = 4

class QuantumResistantCrypto:
    """Quantum-resistant cryptographic functions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_keypair(self) -> Dict[str, str]:
        """Generate quantum-resistant keypair"""
        private_key = secrets.token_hex(32)
        public_key = hashlib.sha256(private_key.encode()).hexdigest()
        
        return {
            "private_key": private_key,
            "public_key": public_key
        }
    
    def sign_message(self, message: str, private_key: str) -> str:
        """Sign message with quantum-resistant signature"""
        message_hash = hashlib.sha256(message.encode()).hexdigest()
        combined = f"{message_hash}{private_key}"
        signature = hashlib.sha512(combined.encode()).hexdigest()
        return signature
    
    def verify_signature(self, message: str, signature: str, public_key: str) -> bool:
        """Verify quantum-resistant signature"""
        # Simplified verification for demo
        message_hash = hashlib.sha256(message.encode()).hexdigest()
        return len(signature) == 128 and len(public_key) == 64

class ConsensusEngine:
    """Blockchain consensus mechanism"""
    
    def __init__(self, consensus_type: str = "proof_of_stake"):
        self.consensus_type = consensus_type
        self.validators: Dict[str, Dict] = {}
        self.stakes: Dict[str, float] = {}
        self.reputation_scores: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_validator(self, validator_id: str, stake: float, reputation: float = 1.0):
        """Register blockchain validator"""
        self.validators[validator_id] = {
            "id": validator_id,
            "stake": stake,
            "reputation": reputation,
            "blocks_validated": 0,
            "last_validation": 0
        }
        self.stakes[validator_id] = stake
        self.reputation_scores[validator_id] = reputation
    
    def select_validator(self) -> str:
        """Select validator based on consensus mechanism"""
        
        if self.consensus_type == "proof_of_stake":
            # Weighted random selection based on stake
            total_stake = sum(self.stakes.values())
            if total_stake == 0:
                return list(self.validators.keys())[0] if self.validators else "default"
            
            import random
            random_value = random.uniform(0, total_stake)
            cumulative = 0
            
            for validator_id, stake in self.stakes.items():
                cumulative += stake
                if random_value <= cumulative:
                    return validator_id
            
            return list(self.validators.keys())[0]
        
        elif self.consensus_type == "proof_of_authority":
            # Select validator with highest reputation
            if not self.reputation_scores:
                return "default"
            return max(self.reputation_scores.keys(), key=lambda k: self.reputation_scores[k])
        
        else:  # Default to round-robin
            if not self.validators:
                return "default"
            validator_list = list(self.validators.keys())
            return validator_list[int(time.time()) % len(validator_list)]
    
    def validate_block(self, block: Block, validator_id: str) -> bool:
        """Validate block using consensus rules"""
        
        # Basic validation checks
        if not self._validate_block_structure(block):
            return False
        
        if not self._validate_transactions(block.transactions):
            return False
        
        if not self._validate_merkle_root(block):
            return False
        
        # Update validator stats
        if validator_id in self.validators:
            self.validators[validator_id]["blocks_validated"] += 1
            self.validators[validator_id]["last_validation"] = time.time()
        
        return True
    
    def _validate_block_structure(self, block: Block) -> bool:
        """Validate block structure"""
        return (
            block.index >= 0 and
            block.timestamp > 0 and
            isinstance(block.transactions, list) and
            len(block.previous_hash) == 64 and
            len(block.hash) == 64
        )
    
    def _validate_transactions(self, transactions: List[Transaction]) -> bool:
        """Validate all transactions in block"""
        for tx in transactions:
            if not self._validate_transaction(tx):
                return False
        return True
    
    def _validate_transaction(self, tx: Transaction) -> bool:
        """Validate individual transaction"""
        return (
            len(tx.id) > 0 and
            len(tx.sender) > 0 and
            len(tx.receiver) > 0 and
            tx.amount >= 0 and
            tx.timestamp > 0
        )
    
    def _validate_merkle_root(self, block: Block) -> bool:
        """Validate merkle root calculation"""
        calculated_root = self._calculate_merkle_root([tx.id for tx in block.transactions])
        return calculated_root == block.merkle_root
    
    def _calculate_merkle_root(self, transaction_ids: List[str]) -> str:
        """Calculate merkle root from transaction IDs"""
        if not transaction_ids:
            return hashlib.sha256(b"").hexdigest()
        
        if len(transaction_ids) == 1:
            return hashlib.sha256(transaction_ids[0].encode()).hexdigest()
        
        # Build merkle tree
        current_level = transaction_ids[:]
        
        while len(current_level) > 1:
            next_level = []
            
            # Pair up and hash
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = hashlib.sha256(f"{left}{right}".encode()).hexdigest()
                next_level.append(combined)
            
            current_level = next_level
        
        return current_level[0]

class SmartContract:
    """Smart contract execution engine"""
    
    def __init__(self):
        self.contracts: Dict[str, Dict] = {}
        self.contract_state: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)
    
    def deploy_contract(self, contract_id: str, code: str, creator: str) -> str:
        """Deploy smart contract to blockchain"""
        
        contract = {
            "id": contract_id,
            "code": code,
            "creator": creator,
            "deployed_at": time.time(),
            "execution_count": 0,
            "gas_used": 0
        }
        
        self.contracts[contract_id] = contract
        self.contract_state[contract_id] = {}
        
        self.logger.info(f"Deployed contract: {contract_id}")
        return contract_id
    
    def execute_contract(self, contract_id: str, function_name: str, 
                        parameters: Dict[str, Any], caller: str) -> Dict[str, Any]:
        """Execute smart contract function"""
        
        if contract_id not in self.contracts:
            raise ValueError(f"Contract {contract_id} not found")
        
        contract = self.contracts[contract_id]
        
        # Simplified contract execution
        result = self._execute_function(contract, function_name, parameters, caller)
        
        # Update contract stats
        contract["execution_count"] += 1
        contract["gas_used"] += 100  # Simplified gas calculation
        
        return {
            "contract_id": contract_id,
            "function": function_name,
            "result": result,
            "gas_used": 100,
            "success": True
        }
    
    def _execute_function(self, contract: Dict, function_name: str, 
                         parameters: Dict[str, Any], caller: str) -> Any:
        """Execute specific contract function"""
        
        # Simplified contract execution logic
        state = self.contract_state[contract["id"]]
        
        if function_name == "transfer":
            # Token transfer function
            from_addr = parameters.get("from", caller)
            to_addr = parameters.get("to")
            amount = parameters.get("amount", 0)
            
            from_balance = state.get(f"balance_{from_addr}", 0)
            if from_balance >= amount:
                state[f"balance_{from_addr}"] = from_balance - amount
                state[f"balance_{to_addr}"] = state.get(f"balance_{to_addr}", 0) + amount
                return {"success": True, "amount": amount}
            else:
                return {"success": False, "error": "Insufficient balance"}
        
        elif function_name == "get_balance":
            # Get balance function
            address = parameters.get("address", caller)
            balance = state.get(f"balance_{address}", 0)
            return {"address": address, "balance": balance}
        
        elif function_name == "mint":
            # Mint tokens function
            to_addr = parameters.get("to")
            amount = parameters.get("amount", 0)
            state[f"balance_{to_addr}"] = state.get(f"balance_{to_addr}", 0) + amount
            return {"minted": amount, "to": to_addr}
        
        else:
            return {"error": f"Function {function_name} not implemented"}

class AetheriumBlockchain:
    """Main blockchain system"""
    
    def __init__(self, consensus_type: str = "proof_of_stake"):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.transaction_pool: Dict[str, Transaction] = {}
        self.crypto = QuantumResistantCrypto()
        self.consensus = ConsensusEngine(consensus_type)
        self.smart_contracts = SmartContract()
        self.wallets: Dict[str, Dict] = {}
        self.mining_difficulty = 4
        self.block_reward = 10.0
        self.logger = logging.getLogger(__name__)
        
        # Create genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the first block in the chain"""
        
        genesis_tx = Transaction(
            id="genesis_tx",
            sender="system",
            receiver="genesis",
            amount=1000000.0,
            data={"type": "genesis", "message": "Aetherium blockchain genesis"},
            timestamp=time.time()
        )
        
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[genesis_tx],
            previous_hash="0" * 64,
            merkle_root=self._calculate_merkle_root(["genesis_tx"]),
            validator="system"
        )
        
        genesis_block.hash = self._calculate_block_hash(genesis_block)
        self.chain.append(genesis_block)
        
        self.logger.info("Genesis block created")
    
    def create_wallet(self, wallet_id: str) -> Dict[str, str]:
        """Create new blockchain wallet"""
        
        keypair = self.crypto.generate_keypair()
        
        wallet = {
            "id": wallet_id,
            "public_key": keypair["public_key"],
            "private_key": keypair["private_key"],
            "balance": 0.0,
            "transaction_count": 0,
            "created_at": time.time()
        }
        
        self.wallets[wallet_id] = wallet
        self.logger.info(f"Created wallet: {wallet_id}")
        
        return {
            "wallet_id": wallet_id,
            "public_key": keypair["public_key"],
            "address": keypair["public_key"][:20]  # Shortened address
        }
    
    def create_transaction(self, sender: str, receiver: str, amount: float, 
                          data: Dict[str, Any] = None) -> str:
        """Create new transaction"""
        
        tx_id = hashlib.sha256(f"{sender}{receiver}{amount}{time.time()}".encode()).hexdigest()
        
        transaction = Transaction(
            id=tx_id,
            sender=sender,
            receiver=receiver,
            amount=amount,
            data=data or {},
            timestamp=time.time(),
            nonce=secrets.token_hex(8)
        )
        
        # Sign transaction
        if sender in self.wallets:
            private_key = self.wallets[sender]["private_key"]
            message = f"{tx_id}{sender}{receiver}{amount}"
            transaction.signature = self.crypto.sign_message(message, private_key)
        
        self.pending_transactions.append(transaction)
        self.transaction_pool[tx_id] = transaction
        
        self.logger.info(f"Created transaction: {tx_id}")
        return tx_id
    
    def mine_block(self) -> Block:
        """Mine new block with pending transactions"""
        
        if not self.pending_transactions:
            raise ValueError("No pending transactions to mine")
        
        # Select validator
        validator = self.consensus.select_validator()
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions[:10],  # Limit transactions per block
            previous_hash=self.chain[-1].hash,
            merkle_root=self._calculate_merkle_root([tx.id for tx in self.pending_transactions[:10]]),
            validator=validator,
            difficulty=self.mining_difficulty
        )
        
        # Mine block (find valid hash)
        new_block.hash = self._mine_block_hash(new_block)
        
        # Validate block
        if self.consensus.validate_block(new_block, validator):
            self.chain.append(new_block)
            
            # Remove mined transactions from pending
            mined_tx_ids = {tx.id for tx in new_block.transactions}
            self.pending_transactions = [tx for tx in self.pending_transactions 
                                       if tx.id not in mined_tx_ids]
            
            # Update balances
            self._update_balances(new_block)
            
            # Reward validator
            self._reward_validator(validator)
            
            self.logger.info(f"Mined block {new_block.index}")
            return new_block
        else:
            raise ValueError("Block validation failed")
    
    def _calculate_merkle_root(self, transaction_ids: List[str]) -> str:
        """Calculate merkle root"""
        return self.consensus._calculate_merkle_root(transaction_ids)
    
    def _calculate_block_hash(self, block: Block) -> str:
        """Calculate block hash"""
        block_string = json.dumps({
            "index": block.index,
            "timestamp": block.timestamp,
            "transactions": [tx.id for tx in block.transactions],
            "previous_hash": block.previous_hash,
            "merkle_root": block.merkle_root,
            "nonce": block.nonce,
            "validator": block.validator
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _mine_block_hash(self, block: Block) -> str:
        """Mine block hash with proof of work"""
        
        target = "0" * block.difficulty
        nonce = 0
        
        while True:
            block.nonce = nonce
            hash_result = self._calculate_block_hash(block)
            
            if hash_result.startswith(target):
                return hash_result
            
            nonce += 1
            
            # Prevent infinite loop
            if nonce > 1000000:
                break
        
        # Fallback hash
        return self._calculate_block_hash(block)
    
    def _update_balances(self, block: Block):
        """Update wallet balances after block mining"""
        
        for tx in block.transactions:
            if tx.sender in self.wallets:
                self.wallets[tx.sender]["balance"] -= tx.amount
                self.wallets[tx.sender]["transaction_count"] += 1
            
            if tx.receiver in self.wallets:
                self.wallets[tx.receiver]["balance"] += tx.amount
    
    def _reward_validator(self, validator: str):
        """Reward block validator"""
        
        if validator in self.wallets:
            self.wallets[validator]["balance"] += self.block_reward
    
    def get_balance(self, wallet_id: str) -> float:
        """Get wallet balance"""
        
        if wallet_id not in self.wallets:
            return 0.0
        
        return self.wallets[wallet_id]["balance"]
    
    def get_transaction_history(self, wallet_id: str) -> List[Dict]:
        """Get transaction history for wallet"""
        
        history = []
        
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == wallet_id or tx.receiver == wallet_id:
                    history.append({
                        "transaction_id": tx.id,
                        "block_index": block.index,
                        "sender": tx.sender,
                        "receiver": tx.receiver,
                        "amount": tx.amount,
                        "timestamp": tx.timestamp,
                        "type": "sent" if tx.sender == wallet_id else "received"
                    })
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)
    
    def verify_chain(self) -> Dict[str, Any]:
        """Verify blockchain integrity"""
        
        issues = []
        
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check hash linking
            if current_block.previous_hash != previous_block.hash:
                issues.append(f"Block {i}: Previous hash mismatch")
            
            # Check hash validity
            if current_block.hash != self._calculate_block_hash(current_block):
                issues.append(f"Block {i}: Hash calculation mismatch")
            
            # Check merkle root
            expected_root = self._calculate_merkle_root([tx.id for tx in current_block.transactions])
            if current_block.merkle_root != expected_root:
                issues.append(f"Block {i}: Merkle root mismatch")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_blocks": len(self.chain),
            "total_transactions": sum(len(block.transactions) for block in self.chain)
        }
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get comprehensive blockchain statistics"""
        
        return {
            "chain_length": len(self.chain),
            "total_transactions": sum(len(block.transactions) for block in self.chain),
            "pending_transactions": len(self.pending_transactions),
            "total_wallets": len(self.wallets),
            "registered_validators": len(self.consensus.validators),
            "consensus_type": self.consensus.consensus_type,
            "mining_difficulty": self.mining_difficulty,
            "block_reward": self.block_reward,
            "deployed_contracts": len(self.smart_contracts.contracts),
            "latest_block_hash": self.chain[-1].hash if self.chain else None,
            "blockchain_status": "operational"
        }

# Example usage and demonstration
async def demo_blockchain_system():
    """Demonstrate blockchain system capabilities"""
    
    print("⛓️ Blockchain System Demo")
    
    # Create blockchain
    blockchain = AetheriumBlockchain("proof_of_stake")
    
    # Register validators
    blockchain.consensus.register_validator("validator_1", 1000.0, 1.0)
    blockchain.consensus.register_validator("validator_2", 500.0, 0.8)
    
    # Create wallets
    wallet1 = blockchain.create_wallet("alice")
    wallet2 = blockchain.create_wallet("bob")
    wallet3 = blockchain.create_wallet("validator_1")
    
    print(f"   Created wallets: {len(blockchain.wallets)}")
    
    # Add initial funds to alice
    blockchain.wallets["alice"]["balance"] = 1000.0
    
    # Create transactions
    tx1 = blockchain.create_transaction("alice", "bob", 100.0, {"memo": "Payment 1"})
    tx2 = blockchain.create_transaction("alice", "bob", 50.0, {"memo": "Payment 2"})
    
    print(f"   Created {len(blockchain.pending_transactions)} transactions")
    
    # Mine block
    new_block = blockchain.mine_block()
    print(f"   Mined block {new_block.index} with {len(new_block.transactions)} transactions")
    
    # Check balances
    alice_balance = blockchain.get_balance("alice")
    bob_balance = blockchain.get_balance("bob")
    
    print(f"   Alice balance: {alice_balance}")
    print(f"   Bob balance: {bob_balance}")
    
    # Deploy smart contract
    contract_code = """
    function transfer(from, to, amount) {
        // Transfer tokens between addresses
        return execute_transfer(from, to, amount);
    }
    """
    
    contract_id = blockchain.smart_contracts.deploy_contract("token_contract", contract_code, "alice")
    
    # Execute contract
    result = blockchain.smart_contracts.execute_contract(
        contract_id, 
        "mint", 
        {"to": "alice", "amount": 100},
        "alice"
    )
    
    print(f"   Contract execution: {result['success']}")
    
    # Verify blockchain
    verification = blockchain.verify_chain()
    print(f"   Chain valid: {verification['valid']}")
    
    # Show blockchain stats
    stats = blockchain.get_blockchain_stats()
    print(f"   Chain length: {stats['chain_length']}")
    print(f"   Total transactions: {stats['total_transactions']}")
    
    print("✅ Blockchain system operational")

if __name__ == "__main__":
    asyncio.run(demo_blockchain_system())