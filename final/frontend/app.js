const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

// Initialize Express app
const app = express();
const port = 3000;

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Landing Page Route
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

// Routes for Face, Object, and Filters
app.get('/face', (req, res) => {
  res.sendFile(path.join(__dirname, 'views', 'face.html'));
});

app.get('/object', (req, res) => {
  res.sendFile(path.join(__dirname, 'views', 'object.html'));
});

app.get('/filters', (req, res) => {
  res.sendFile(path.join(__dirname, 'views', 'filters.html'));
});

// Proxy routes to Flask backend
const flaskBackendUrl = 'http://localhost:5000';
app.use('/api/face-analysis', createProxyMiddleware({ target: flaskBackendUrl, changeOrigin: true }));
app.use('/api/object-detection', createProxyMiddleware({ target: flaskBackendUrl, changeOrigin: true }));
app.use('/api/apply-filters', createProxyMiddleware({ target: flaskBackendUrl, changeOrigin: true }));

// Start server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
