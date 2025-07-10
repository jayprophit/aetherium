// Shared types for ERP module (integrated with knowledge-base)

export interface User {
  id: string;
  name: string;
  email: string;
  roles: string[];
}

export interface LedgerEntry {
  id: string;
  date: string;
  amount: number;
  description: string;
  account: string;
}
