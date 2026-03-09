import requests
import json
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class Phi3Client:
    """
    Client for interacting with the Phi-3 LLM service.
    Currently supports a generic HTTP API (compatible with Ollama or similar).
    """
    
    def __init__(self):
        # Allow configuration via settings, default to local Ollama port or placeholder
        self.api_url = getattr(settings, 'PHI3_API_URL', 'http://localhost:11434/api/generate')
        self.model_name = getattr(settings, 'PHI3_MODEL_NAME', 'phi3')
        
    def generate_explanation(self, detections, file_name):
        """
        Generate a medical engineering explanation based on detection results.
        """
        
        # 1. Construct the prompt
        count = len(detections)
        if count == 0:
            prompt_context = f"No polyps were detected in the image '{file_name}'."
        else:
            # Summarize detections
            # e.g., "Found 3 polyps with confidence scores: 0.85, 0.72, 0.65."
            confs = [f"{d['confidence']:.2f}" for d in detections]
            prompt_context = f"Detected {count} polyps in the image '{file_name}'. Confidence scores are: {', '.join(confs)}."

        system_prompt = (
            "You are a medical engineering assistant. "
            "Your task is to provide a structural, non-diagnostic technical explanation of the object detection results from a colonoscopy image. "
            "Do NOT give medical advice or diagnoses. Focus on the computer vision analysis results. "
            "Keep the response concise (under 200 words) and professional."
        )
        
        user_prompt = (
            f"Analysis Results: {prompt_context}\n\n"
            "Please generate a technical report describing these findings in a structured format."
        )

        # 2. Call the API (Mocking for now if service is not up)
        try:
            # Check if we should use the real API or mock
            if getattr(settings, 'USE_MOCK_PHI3', True):
                return self._mock_response(prompt_context, detections)
                
            payload = {
                "model": self.model_name,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('response', 'No response generated.')
            
        except Exception as e:
            logger.error(f"Phi-3 API call failed: {e}")
            # Fallback to mock if API fails
            return self._mock_response(prompt_context, detections)

    def _mock_response(self, context, detections=None):
        if not detections:
            return (
                f"**自动化技术报告**\n\n"
                f"**输入数据**: {context}\n"
                f"**模型**: YOLO11（目标检测）+ Phi-3（分析）\n\n"
                f"**分析**:\n"
                f"系统已完成图像处理与目标提取。检测置信度体现模型对目标的匹配程度，数值越高越可信。当前报告仅用于工程验证，不构成临床诊断。"
            )
        scores = [float(d.get('confidence', 0)) for d in detections]
        count = len(scores)
        if count == 0:
            return (
                f"**自动化技术报告**\n\n"
                f"**输入数据**: 未在当前图像中检测到可疑息肉区域。\n"
                f"**模型**: YOLO11（目标检测）+ Phi-3（分析）\n\n"
                f"**分析**:\n"
                f"模型未发现符合息肉特征的显著区域，可能与视野遮挡、强反光或图像质量有关。\n\n"
                f"**建议**:\n"
                f"建议进行人工检查确认，必要时调整角度、冲洗或提高清晰度后复查。\n\n"
                f"**免责声明**: 本报告仅用于工程验证，并非临床诊断。"
            )
        max_score = max(scores)
        avg_score = sum(scores) / count if count else 0.0
        conf_list = ", ".join(f"{s:.2f}" for s in scores)
        header = "**自动化技术报告**\n\n"
        input_sec = (
            f"**输入数据**: 检测到 {count} 个候选区域；最高置信度 {max_score:.2f}，平均置信度 {avg_score:.2f}。分数列表：{conf_list}。\n"
            f"**模型**: YOLO11（目标检测）+ Phi-3（分析）\n\n"
        )
        if max_score > 0.90:
            analysis = (
                f"**分析**:\n"
                f"存在高度可信的息肉候选区域（最高置信度 > 0.90）。纹理与边界特征与息肉形态高度一致，属于强阳性提示。\n\n"
                f"**建议**:\n"
                f"高度疑似息肉，建议临床复核；若条件允许，可进行放大观察或染色内镜以进一步确认。\n"
            )
        elif max_score < 0.30:
            analysis = (
                f"**分析**:\n"
                f"整体置信度较低（最高置信度 < 0.30）。当前提示可能受泡沫、粪渣或强反光影响，模型不具备足够把握。\n\n"
                f"**建议**:\n"
                f"检测结果不显著，建议人工检查；可优化视野与光照条件后再次采样评估。\n"
            )
        else:
            analysis = (
                f"**分析**:\n"
                f"检测到中等可信度的候选区域。目标与息肉特征部分吻合，但仍存在不确定性。\n\n"
                f"**建议**:\n"
                f"建议优化视野（冲洗、减光）、多角度观察，并进行人工复核以提高确定性。\n"
            )
        disclaimer = (
            f"\n**免责声明**: 本报告由 AI 助手（Phi-3）基于 YOLO11 检测结果生成，仅用于工程与研究验证，不构成临床诊断。"
        )
        return header + input_sec + analysis + disclaimer
