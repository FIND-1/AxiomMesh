import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './Stage1.css';

export default function Stage1({ responses }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

  const modelLabel = (model) => model?.split('/')[1] || model || 'unknown';

  return (
    <div className="stage stage1">
      <h3 className="stage-title">阶段一：专家分析</h3>

      <div className="tabs">
        {responses.map((resp, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            {resp.agent_name || 'Agent'} · {modelLabel(resp.model)}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="model-name">
          {(responses[activeTab].agent_name || 'Agent')} · {responses[activeTab].model}
        </div>
        <div className="response-text markdown-content">
          <ReactMarkdown>{responses[activeTab].response}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
