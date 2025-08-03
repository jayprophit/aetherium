// ERP CRM Service (Integrated)
import express from 'express';
const app = express();

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

// TODO: Implement CRM APIs (sales, customer service)

export default app;
