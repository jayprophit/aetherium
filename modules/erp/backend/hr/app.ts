// ERP HR Service (Integrated)
import express from 'express';
const app = express();

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

// TODO: Implement HR APIs (payroll, recruitment, performance)

export default app;
