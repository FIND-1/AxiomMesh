import ReactMarkdown from 'react-markdown';
import './Stage2.css';

export default function Stage2({ evaluation }) {
  if (!evaluation) {
    return null;
  }

  const structured = evaluation.structured_output || {};
  const scorecard = structured.scorecard || [];
  const evidence = structured.evidence || structured.supporting_evidence || [];
  const risks = structured.gaps || [];
  const recommendedActions = structured.recommendations || structured.recommended_actions || [];

  return (
    <div className="stage stage2">
      <h3 className="stage-title">阶段二：裁判评审</h3>

      <div className="ranking-model">
        {(evaluation.agent_name || 'Judge Agent')} · {evaluation.model}
      </div>

      <div className="ranking-content markdown-content">
        <ReactMarkdown>{evaluation.response}</ReactMarkdown>
      </div>

      {scorecard.length > 0 && (
        <div className="aggregate-rankings">
          <h4>Agent Scorecard</h4>
          <p className="stage-description">
            Judge 会从证据质量、推理质量和行动价值三个维度评估每个 Agent。
          </p>
          <div className="aggregate-list">
            {scorecard.map((item, index) => (
              <div key={index} className="aggregate-item">
                <span className="rank-model">{item.agent_name}</span>
                <span className="rank-score">证据: {item.evidence_score}/5</span>
                <span className="rank-score">推理: {item.reasoning_score}/5</span>
                <span className="rank-score">行动: {item.actionability_score}/5</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {evidence.length > 0 && (
        <div className="parsed-ranking">
          <strong>支持证据</strong>
          <ol>
            {evidence.map((item, index) => (
              <li key={index}>
                [{item.credibility}] {item.detail}
              </li>
            ))}
          </ol>
        </div>
      )}

      {risks.length > 0 && (
        <div className="parsed-ranking">
          <strong>剩余风险</strong>
          <ol>
            {risks.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ol>
        </div>
      )}

      {recommendedActions.length > 0 && (
        <div className="parsed-ranking">
          <strong>建议行动</strong>
          <ol>
            {recommendedActions.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
