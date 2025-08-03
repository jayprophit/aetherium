// ERP Finance Service (Integrated)
import express from 'express';
const app = express();

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

// TODO: Implement finance APIs (ledger, invoicing, reporting)

export default app;
