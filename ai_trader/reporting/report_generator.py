"""
간단한 HTML 보고서 생성기

훈련 메트릭과 에피소드 정보를 HTML 보고서로 생성합니다.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """HTML 보고서 생성 클래스"""
    
    @staticmethod
    def generate_embedding_report(
        output_path: str,
        training_metadata: Dict[str, Any],
        final_metrics: Dict[str, float],
        config: Dict[str, Any],
        tensorboard_dir: Optional[str] = None
    ) -> str:
        """임베딩 모델 훈련 보고서 생성"""
        logger.info(f"Generating embedding report: {output_path}")
        
        html = HTMLReportGenerator._create_html_template(
            title="임베딩 모델 훈련 보고서",
            content=HTMLReportGenerator._build_embedding_content(
                training_metadata, final_metrics, config, tensorboard_dir
            )
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Report saved: {output_path}")
        return str(output_path)
    
    @staticmethod
    def generate_grpo_report(
        output_path: str,
        training_metadata: Dict[str, Any],
        final_metrics: Dict[str, float],
        config: Dict[str, Any],
        sample_episodes: Optional[List[Dict[str, Any]]] = None,
        tensorboard_dir: Optional[str] = None
    ) -> str:
        """GRPO 훈련 보고서 생성"""
        logger.info(f"Generating GRPO report: {output_path}")
        
        html = HTMLReportGenerator._create_html_template(
            title="GRPO 훈련 보고서",
            content=HTMLReportGenerator._build_grpo_content(
                training_metadata, final_metrics, config, sample_episodes, tensorboard_dir
            )
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Report saved: {output_path}")
        return str(output_path)
    
    @staticmethod
    def _create_html_template(title: str, content: str) -> str:
        """HTML 템플릿 생성"""
        return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }}
        .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f8f9fa; padding: 20px; border-radius: 5px; border-left: 4px solid #3498db; }}
        .metric.success {{ border-left-color: #27ae60; }}
        .metric.warning {{ border-left-color: #f39c12; }}
        .metric.danger {{ border-left-color: #e74c3c; }}
        .metric-label {{ font-size: 0.9em; color: #7f8c8d; }}
        .metric-value {{ font-size: 1.8em; font-weight: bold; color: #2c3e50; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
        .episode {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #95a5a6; }}
        .episode.profit {{ border-left-color: #27ae60; }}
        .episode.loss {{ border-left-color: #e74c3c; }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ecf0f1; text-align: center; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        {content}
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>AI Trader - GRPO Scalping System</p>
        </div>
    </div>
</body>
</html>"""
    
    @staticmethod
    def _build_embedding_content(
        training_metadata: Dict[str, Any],
        final_metrics: Dict[str, float],
        config: Dict[str, Any],
        tensorboard_dir: Optional[str]
    ) -> str:
        """임베딩 보고서 내용 생성"""
        content = "<h1>📊 임베딩 모델 훈련 보고서</h1>"
        
        # 메타데이터
        content += '<div class="metadata"><h3>훈련 정보</h3>'
        content += f'<p><strong>완료 시간:</strong> {training_metadata.get("timestamp", "N/A")}</p>'
        content += f'<p><strong>에포크:</strong> {training_metadata.get("epoch", "N/A")}</p>'
        content += f'<p><strong>임베딩 차원:</strong> {config.get("embedding_dim", "N/A")}</p>'
        content += f'<p><strong>시퀀스 길이:</strong> {config.get("seq_len", "N/A")}</p>'
        if tensorboard_dir:
            content += f'<p><strong>TensorBoard:</strong> {tensorboard_dir}</p>'
        content += '</div>'
        
        # 메트릭
        content += '<h2>최종 메트릭</h2><div class="metrics">'
        
        val_loss = final_metrics.get('val_loss', 0.0)
        loss_status = 'success' if val_loss < 0.5 else 'warning' if val_loss < 1.0 else 'danger'
        content += f'<div class="metric {loss_status}"><div class="metric-label">검증 손실</div>'
        content += f'<div class="metric-value">{val_loss:.4f}</div></div>'
        
        silhouette = final_metrics.get('silhouette_score', 0.0)
        sil_status = 'success' if silhouette > 0.5 else 'warning' if silhouette > 0.3 else 'danger'
        content += f'<div class="metric {sil_status}"><div class="metric-label">Silhouette Score</div>'
        content += f'<div class="metric-value">{silhouette:.4f}</div></div>'
        
        temporal = final_metrics.get('temporal_coherence', 0.0)
        temp_status = 'success' if temporal > 0.7 else 'warning' if temporal > 0.5 else 'danger'
        content += f'<div class="metric {temp_status}"><div class="metric-label">Temporal Coherence</div>'
        content += f'<div class="metric-value">{temporal:.4f}</div></div>'
        
        content += '</div>'
        
        # 설정
        content += '<h2>모델 설정</h2><table><tr><th>파라미터</th><th>값</th></tr>'
        for key, value in config.items():
            content += f'<tr><td>{key}</td><td>{value}</td></tr>'
        content += '</table>'
        
        return content
    
    @staticmethod
    def _build_grpo_content(
        training_metadata: Dict[str, Any],
        final_metrics: Dict[str, float],
        config: Dict[str, Any],
        sample_episodes: Optional[List[Dict[str, Any]]],
        tensorboard_dir: Optional[str]
    ) -> str:
        """GRPO 보고서 내용 생성"""
        content = "<h1>🤖 GRPO 스캘핑 에이전트 훈련 보고서</h1>"
        
        # 메타데이터
        content += '<div class="metadata"><h3>훈련 정보</h3>'
        content += f'<p><strong>완료 시간:</strong> {training_metadata.get("timestamp", "N/A")}</p>'
        content += f'<p><strong>총 타임스텝:</strong> {training_metadata.get("total_timesteps", 0):,}</p>'
        content += f'<p><strong>총 업데이트:</strong> {training_metadata.get("num_updates", 0):,}</p>'
        content += f'<p><strong>빠른 손절 임계값:</strong> {training_metadata.get("quick_exit_threshold", "N/A")}초</p>'
        if tensorboard_dir:
            content += f'<p><strong>TensorBoard:</strong> {tensorboard_dir}</p>'
        content += '</div>'
        
        # 메트릭
        content += '<h2>최종 성능 메트릭</h2><div class="metrics">'
        
        win_rate = final_metrics.get('win_rate', 0.0)
        win_status = 'success' if win_rate > 0.55 else 'warning' if win_rate > 0.50 else 'danger'
        content += f'<div class="metric {win_status}"><div class="metric-label">승률</div>'
        content += f'<div class="metric-value">{win_rate * 100:.1f}%</div></div>'
        
        sharpe = final_metrics.get('sharpe_ratio', 0.0)
        sharpe_status = 'success' if sharpe > 1.5 else 'warning' if sharpe > 1.0 else 'danger'
        content += f'<div class="metric {sharpe_status}"><div class="metric-label">샤프 비율</div>'
        content += f'<div class="metric-value">{sharpe:.2f}</div></div>'
        
        holding_time = final_metrics.get('avg_holding_time', 0.0)
        holding_status = 'success' if holding_time < 30 else 'warning' if holding_time < 60 else 'danger'
        content += f'<div class="metric {holding_status}"><div class="metric-label">평균 보유 시간</div>'
        content += f'<div class="metric-value">{holding_time:.1f}초</div></div>'
        
        avg_profit = final_metrics.get('avg_profit_per_trade', 0.0)
        profit_status = 'success' if avg_profit > 0.005 else 'warning' if avg_profit > 0 else 'danger'
        content += f'<div class="metric {profit_status}"><div class="metric-label">거래당 평균 수익</div>'
        content += f'<div class="metric-value">{avg_profit * 100:.2f}%</div></div>'
        
        content += '</div>'
        
        # 샘플 에피소드
        if sample_episodes:
            content += '<h2>샘플 거래 에피소드</h2>'
            for i, ep in enumerate(sample_episodes[:10], 1):
                profit = ep.get('total_profit', 0.0)
                ep_class = 'profit' if profit > 0 else 'loss'
                profit_sign = '+' if profit > 0 else ''
                profit_class = 'positive' if profit > 0 else 'negative'
                
                content += f'<div class="episode {ep_class}"><h3>에피소드 #{i}</h3>'
                content += f'<p><strong>총 수익:</strong> <span class="{profit_class}">{profit_sign}{profit * 100:.2f}%</span></p>'
                content += f'<p><strong>거래 횟수:</strong> {ep.get("num_trades", 0)}</p>'
                content += f'<p><strong>평균 보유 시간:</strong> {ep.get("avg_holding_time", 0.0):.1f}초</p>'
                content += f'<p><strong>샤프 비율:</strong> {ep.get("sharpe_ratio", 0.0):.2f}</p>'
                content += '</div>'
        
        # 설정
        content += '<h2>하이퍼파라미터</h2><table><tr><th>파라미터</th><th>값</th></tr>'
        for key, value in config.items():
            content += f'<tr><td>{key}</td><td>{value}</td></tr>'
        content += '</table>'
        
        return content


# 편의 함수
def generate_embedding_report(*args, **kwargs):
    """임베딩 보고서 생성 (편의 함수)"""
    return HTMLReportGenerator.generate_embedding_report(*args, **kwargs)


def generate_grpo_report(*args, **kwargs):
    """GRPO 보고서 생성 (편의 함수)"""
    return HTMLReportGenerator.generate_grpo_report(*args, **kwargs)


def generate_combined_report(
    output_path: str,
    embedding_metadata: Dict[str, Any],
    embedding_metrics: Dict[str, float],
    grpo_metadata: Dict[str, Any],
    grpo_metrics: Dict[str, float],
    sample_episodes: Optional[List[Dict[str, Any]]] = None
) -> str:
    """통합 보고서 생성"""
    logger.info(f"Generating combined report: {output_path}")
    
    content = "<h1>📊 GRPO 스캘핑 시스템 통합 훈련 보고서</h1>"
    
    # 임베딩 섹션
    content += '<h2>1️⃣ 임베딩 모델</h2><div class="metadata">'
    content += f'<p><strong>완료:</strong> {embedding_metadata.get("timestamp", "N/A")}</p>'
    content += f'<p><strong>에포크:</strong> {embedding_metadata.get("epoch", "N/A")}</p>'
    content += '</div><div class="metrics">'
    
    val_loss = embedding_metrics.get('val_loss', 0.0)
    content += f'<div class="metric"><div class="metric-label">검증 손실</div>'
    content += f'<div class="metric-value">{val_loss:.4f}</div></div>'
    
    silhouette = embedding_metrics.get('silhouette_score', 0.0)
    content += f'<div class="metric"><div class="metric-label">Silhouette Score</div>'
    content += f'<div class="metric-value">{silhouette:.4f}</div></div>'
    
    content += '</div>'
    
    # GRPO 섹션
    content += '<h2>2️⃣ GRPO 에이전트</h2><div class="metadata">'
    content += f'<p><strong>완료:</strong> {grpo_metadata.get("timestamp", "N/A")}</p>'
    content += f'<p><strong>타임스텝:</strong> {grpo_metadata.get("total_timesteps", 0):,}</p>'
    content += '</div><div class="metrics">'
    
    win_rate = grpo_metrics.get('win_rate', 0.0)
    content += f'<div class="metric"><div class="metric-label">승률</div>'
    content += f'<div class="metric-value">{win_rate * 100:.1f}%</div></div>'
    
    sharpe = grpo_metrics.get('sharpe_ratio', 0.0)
    content += f'<div class="metric"><div class="metric-label">샤프 비율</div>'
    content += f'<div class="metric-value">{sharpe:.2f}</div></div>'
    
    content += '</div>'
    
    html = HTMLReportGenerator._create_html_template("통합 훈련 보고서", content)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"Report saved: {output_path}")
    return str(output_path)
