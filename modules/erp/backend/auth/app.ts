// ERP Auth Service (Integrated)
import express from 'express';
const app = express();

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

// TODO: Implement OAuth2, RBAC, JWT endpoints

export default app;
