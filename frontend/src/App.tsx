
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ZustandProvider } from './app/providers/ZustandProvider';
import { ThemeProvider } from './app/providers/ThemeProvider';
import AgentDashboard from './pages/AgentDashboard/index';
import LearningAnalyticsPage from './pages/LearningAnalytics/LearningAnalyticsPage';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ZustandProvider>
        <ThemeProvider>
          <Router basename="/ai-agent-meta-cognitive">
            <div className="App">
              <nav className="bg-gray-800 text-white p-4">
                <div className="max-w-7xl mx-auto flex space-x-4">
                  <Link to="/" className="px-3 py-2 rounded-md hover:bg-gray-700">Агент</Link>
                  <Link to="/learning-analytics" className="px-3 py-2 rounded-md hover:bg-gray-700">Аналитика обучения</Link>
                </div>
              </nav>
              <Routes>
                <Route path="/" element={<AgentDashboard />} />
                <Route path="/agent-dashboard" element={<AgentDashboard />} />
                <Route path="/learning-analytics" element={<LearningAnalyticsPage />} />
              </Routes>
            </div>
          </Router>
        </ThemeProvider>
      </ZustandProvider>
    </QueryClientProvider>
  );
}

export default App;
