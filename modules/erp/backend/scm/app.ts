// ERP SCM Service (Integrated)
import express from 'express';
const app = express();

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

// TODO: Implement SCM APIs (inventory, procurement, logistics)

export default app;
