# -*- coding: utf-8 -*-
"""
综合评分模块
整合三个评分维度，计算代币项目的综合风险评分

评分权重:
- 独立 EOA 分析: 40分 (活跃度/真实用户)
- 持有者集中度: 30分 (抛压风险)
- 合约权限: 30分 (Rug Pull 风险)

总分: 100分

输出格式专为前端展示设计

数据源:
- fast: Nansen API (推荐，速度快，带 Smart Money 标签)
- deep: BlockPi RPC (完整分析)
- auto: 自动选择 (有 Nansen 则用 fast)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.blockchain.web3_client import Web3Client
from src.blockchain.nansen_client import NansenClient, NansenError
from src.scoring.unique_eoa import UniqueEOAAnalyzer
from src.scoring.holder_analysis import HolderAnalyzer
from src.scoring.contract_permission import ContractPermissionAnalyzer


# 风险等级配置 (前端可用于显示颜色)
RISK_LEVEL_CONFIG = {
    "low_risk": {
        "label": "Low Risk",
        "label_cn": "低风险",
        "color": "#22c55e",      # green
        "bg_color": "#dcfce7",
        "icon": "shield-check"
    },
    "medium_risk": {
        "label": "Medium Risk",
        "label_cn": "中等风险",
        "color": "#eab308",      # yellow
        "bg_color": "#fef9c3",
        "icon": "alert-triangle"
    },
    "high_risk": {
        "label": "High Risk",
        "label_cn": "高风险",
        "color": "#f97316",      # orange
        "bg_color": "#ffedd5",
        "icon": "alert-circle"
    },
    "extreme_risk": {
        "label": "Extreme Risk",
        "label_cn": "极高风险",
        "color": "#ef4444",      # red
        "bg_color": "#fee2e2",
        "icon": "x-circle"
    },
    "unknown": {
        "label": "Unknown",
        "label_cn": "未知",
        "color": "#6b7280",      # gray
        "bg_color": "#f3f4f6",
        "icon": "help-circle"
    }
}

# 风险标签配置 (前端可用于显示 badge)
RISK_TAGS_CONFIG = {
    # EOA 相关
    "ORGANIC_GROWTH": {
        "label": "Organic Growth",
        "label_cn": "真实用户增长",
        "type": "success",
        "category": "activity"
    },
    "MODERATE_ACTIVITY": {
        "label": "Moderate Activity",
        "label_cn": "中等活跃",
        "type": "warning",
        "category": "activity"
    },
    "LOW_ACTIVITY": {
        "label": "Low Activity",
        "label_cn": "低活跃/疑似刷量",
        "type": "danger",
        "category": "activity"
    },
    # 持有者相关
    "DISTRIBUTED": {
        "label": "Well Distributed",
        "label_cn": "持仓分散",
        "type": "success",
        "category": "holder"
    },
    "CONCENTRATED": {
        "label": "Concentrated",
        "label_cn": "持仓集中",
        "type": "warning",
        "category": "holder"
    },
    "WHALE_CONTROLLED": {
        "label": "Whale Controlled",
        "label_cn": "大户控盘",
        "type": "danger",
        "category": "holder"
    },
    "EXTREME_CONCENTRATION": {
        "label": "Extreme Concentration",
        "label_cn": "极端集中",
        "type": "danger",
        "category": "holder"
    },
    # 合约权限相关
    "SAFE_CONTRACT": {
        "label": "Safe Contract",
        "label_cn": "安全合约",
        "type": "success",
        "category": "permission"
    },
    "LIMITED_RISK": {
        "label": "Limited Risk",
        "label_cn": "有限风险",
        "type": "warning",
        "category": "permission"
    },
    "RUG_RISK": {
        "label": "Rug Risk",
        "label_cn": "跑路风险",
        "type": "danger",
        "category": "permission"
    }
}


class TotalScorer:
    """
    综合评分器

    支持两种数据源:
    - fast: 使用 Nansen API (推荐，速度快，带 Smart Money 标签)
    - deep: 使用 BlockPi RPC (完整分析)
    - auto: 自动选择 (有 Nansen 则用 fast)

    使用示例:
        # 快速模式 (推荐)
        scorer = TotalScorer(web3_client, nansen_client)
        result = scorer.score_token(token_address, mode="fast")

        # 深度模式
        result = scorer.score_token(token_address, mode="deep")
    """

    def __init__(
        self,
        client: Web3Client,
        nansen: Optional[NansenClient] = None,
        use_cache: bool = True
    ):
        """
        初始化综合评分器

        Args:
            client: Web3 客户端实例 (BlockPi RPC)
            nansen: Nansen 客户端实例 (可选，用于快速模式)
            use_cache: 是否使用缓存
        """
        self.client = client
        self.nansen = nansen
        self.use_cache = use_cache

        # 初始化三个分析器 (传入 Nansen 客户端)
        self.eoa_analyzer = UniqueEOAAnalyzer(client, nansen=nansen, use_cache=use_cache)
        self.holder_analyzer = HolderAnalyzer(client, nansen=nansen, use_cache=use_cache)
        self.permission_analyzer = ContractPermissionAnalyzer(client, use_cache)

    def score_token(
        self,
        token_address: str,
        mode: str = "auto",
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        time_window_hours: int = 24,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        对代币进行综合评分

        Args:
            token_address: 代币合约地址
            mode: 分析模式
                - "auto": 自动选择 (有 Nansen 则用 fast)
                - "fast": 使用 Nansen API (推荐)
                - "deep": 使用 BlockPi RPC (完整分析)
            from_block: 起始区块 (仅 deep 模式, None = 自动计算)
            to_block: 结束区块 (仅 deep 模式, None = 最新区块)
            time_window_hours: EOA 分析时间窗口 (小时)
            limit: 分析的交易数量上限 (仅 fast 模式, 默认 1000)

        Returns:
            前端友好的综合评分结果字典
        """
        # 自动选择模式
        if mode == "auto":
            mode = "fast" if self.nansen else "deep"

        print(f"\n{'='*60}")
        print(f"  Token Scoring: {token_address}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'='*60}")

        # 获取区块范围 (仅 deep 模式需要)
        if mode == "deep":
            if to_block is None:
                to_block = self.client.get_block_number()
            if from_block is None:
                from_block = max(0, to_block - 10000)
            print(f"\nBlock range: {from_block} -> {to_block}")

        # 1. EOA 分析 (40分)
        print(f"\n[1/3] Analyzing unique EOA...")
        eoa_result = self._analyze_eoa(token_address, mode, from_block, to_block, time_window_hours, limit)

        # 2. 持有者分析 (30分)
        print(f"\n[2/3] Analyzing holder concentration...")
        holder_result = self._analyze_holders(token_address, mode, from_block, to_block)

        # 3. 合约权限分析 (30分) - 不受 mode 影响
        print(f"\n[3/3] Analyzing contract permissions...")
        permission_result = self._analyze_permissions(token_address)

        # 4. 计算综合评分
        total_score = self._calculate_total_score(eoa_result, holder_result, permission_result)
        risk_level = self._determine_risk_level(total_score)

        # 5. 生成风险标签
        risk_tags = self._generate_risk_tags(eoa_result, holder_result, permission_result)

        # 6. 组装前端友好的结果
        result = self._build_frontend_response(
            token_address=token_address,
            mode=mode,
            from_block=from_block,
            to_block=to_block,
            total_score=total_score,
            risk_level=risk_level,
            risk_tags=risk_tags,
            eoa_result=eoa_result,
            holder_result=holder_result,
            permission_result=permission_result
        )

        self._print_summary(result)

        return result

    def _analyze_eoa(
        self,
        token_address: str,
        mode: str,
        from_block: Optional[int],
        to_block: Optional[int],
        time_window_hours: int,
        limit: int
    ) -> Dict:
        """执行 EOA 分析"""
        try:
            return self.eoa_analyzer.analyze(
                token_address,
                mode=mode,
                from_block=from_block or 0,
                to_block=to_block,
                time_window_hours=time_window_hours,
                limit=limit
            )
        except Exception as e:
            print(f"  EOA analysis failed: {e}")
            return {"score": 0, "max_score": 40.0, "risk_level": "unknown", "error": str(e), "data_source": "error"}

    def _analyze_holders(
        self,
        token_address: str,
        mode: str,
        from_block: Optional[int],
        to_block: Optional[int]
    ) -> Dict:
        """执行持有者分析"""
        try:
            return self.holder_analyzer.analyze(
                token_address,
                mode=mode,
                from_block=from_block or 0,
                to_block=to_block
            )
        except Exception as e:
            print(f"  Holder analysis failed: {e}")
            return {"score": 0, "max_score": 30.0, "risk_level": "unknown", "error": str(e), "data_source": "error"}

    def _analyze_permissions(self, token_address: str) -> Dict:
        """执行合约权限分析"""
        try:
            return self.permission_analyzer.analyze_contract(token_address)
        except Exception as e:
            print(f"  Permission analysis failed: {e}")
            return {"score": 0, "risk_level": "unknown", "error": str(e)}

    def _calculate_total_score(self, eoa_result: Dict, holder_result: Dict, permission_result: Dict) -> float:
        """计算综合评分"""
        eoa_score = eoa_result.get("score", 0)
        holder_score = holder_result.get("score", 0)
        permission_score = permission_result.get("score", 0)
        return round(eoa_score + holder_score + permission_score, 2)

    def _determine_risk_level(self, total_score: float) -> str:
        """确定风险等级"""
        if total_score >= 80:
            return "low_risk"
        elif total_score >= 60:
            return "medium_risk"
        elif total_score >= 40:
            return "high_risk"
        else:
            return "extreme_risk"

    def _generate_risk_tags(self, eoa_result: Dict, holder_result: Dict, permission_result: Dict) -> List[str]:
        """生成风险标签"""
        tags = []

        # EOA 标签
        eoa_level = eoa_result.get("risk_level", "unknown")
        if eoa_level == "low_risk":
            tags.append("ORGANIC_GROWTH")
        elif eoa_level == "medium_risk":
            tags.append("MODERATE_ACTIVITY")
        elif eoa_level == "high_risk":
            tags.append("LOW_ACTIVITY")

        # 持有者标签
        holder_level = holder_result.get("risk_level", "unknown")
        if holder_level == "low_risk":
            tags.append("DISTRIBUTED")
        elif holder_level == "medium_risk":
            tags.append("CONCENTRATED")
        elif holder_level == "high_risk":
            tags.append("WHALE_CONTROLLED")
        elif holder_level == "extreme_risk":
            tags.append("EXTREME_CONCENTRATION")

        # 权限标签
        permission_level = permission_result.get("risk_level", "unknown")
        if permission_level == "low_risk":
            tags.append("SAFE_CONTRACT")
        elif permission_level == "medium_risk":
            tags.append("LIMITED_RISK")
        elif permission_level == "high_risk":
            tags.append("RUG_RISK")

        return tags

    def _build_frontend_response(
        self,
        token_address: str,
        mode: str,
        from_block: Optional[int],
        to_block: Optional[int],
        total_score: float,
        risk_level: str,
        risk_tags: List[str],
        eoa_result: Dict,
        holder_result: Dict,
        permission_result: Dict
    ) -> Dict[str, Any]:
        """
        构建前端友好的响应格式

        设计原则:
        1. 顶层是概览信息，供首页/卡片展示
        2. scores 包含三个维度详情，供详情页展示
        3. 所有标签/等级都带有前端渲染需要的配置
        4. data_source 字段表明数据来源
        """
        risk_config = RISK_LEVEL_CONFIG.get(risk_level, RISK_LEVEL_CONFIG["unknown"])

        return {
            # === 基本信息 ===
            "token_address": token_address,
            "timestamp": datetime.now().isoformat(),
            "analysis_mode": mode,
            "block_range": {
                "from": from_block,
                "to": to_block
            } if mode == "deep" else None,

            # === 数据源信息 ===
            "data_sources": {
                "eoa": eoa_result.get("data_source", "unknown"),
                "holder": holder_result.get("data_source", "unknown"),
                "permission": "blockpi"  # 权限分析始终使用 RPC
            },

            # === 概览 (首页/卡片展示) ===
            "overview": {
                "total_score": total_score,
                "max_score": 100,
                "risk_level": risk_level,
                "risk_label": risk_config["label"],
                "risk_label_cn": risk_config["label_cn"],
                "risk_color": risk_config["color"],
                "risk_bg_color": risk_config["bg_color"],
                "risk_icon": risk_config["icon"]
            },

            # === 风险标签 (badge 展示) ===
            "risk_tags": [
                {
                    "key": tag,
                    "label": RISK_TAGS_CONFIG[tag]["label"],
                    "label_cn": RISK_TAGS_CONFIG[tag]["label_cn"],
                    "type": RISK_TAGS_CONFIG[tag]["type"],
                    "category": RISK_TAGS_CONFIG[tag]["category"]
                }
                for tag in risk_tags if tag in RISK_TAGS_CONFIG
            ],

            # === 分项评分 (详情页展示) ===
            "scores": {
                # EOA 活跃度分析
                "eoa": {
                    "name": "User Activity",
                    "name_cn": "用户活跃度",
                    "description": "Unique EOA analysis to detect fake activity",
                    "description_cn": "独立EOA分析，检测虚假活跃",
                    "score": eoa_result.get("score", 0),
                    "max_score": 40,
                    "weight": "40%",
                    "risk_level": eoa_result.get("risk_level", "unknown"),
                    "metrics": {
                        "unique_eoa_count": eoa_result.get("unique_eoa_count", 0),
                        "total_addresses": eoa_result.get("total_addresses", 0),
                        "eoa_percentage": round(eoa_result.get("eoa_percentage", 0), 2),
                        "events_count": eoa_result.get("events_count", 0),
                        "smart_money_count": eoa_result.get("smart_money_count", 0),
                        "dex_pool_count": eoa_result.get("dex_pool_count", 0),
                        "bot_count": eoa_result.get("bot_count", 0),
                        "user_types": eoa_result.get("user_types", {})
                    }
                },
                # 持有者集中度分析
                "holder": {
                    "name": "Holder Distribution",
                    "name_cn": "持仓分布",
                    "description": "Top holder concentration analysis",
                    "description_cn": "Top持有者集中度分析，评估抛压风险",
                    "score": holder_result.get("score", 0),
                    "max_score": 30,
                    "weight": "30%",
                    "risk_level": holder_result.get("risk_level", "unknown"),
                    "metrics": {
                        "total_holders": holder_result.get("total_holders", 0),
                        "top10_percentage": round(holder_result.get("top10_percentage", 0), 2),
                        "top10_holders": self._format_top_holders(holder_result.get("top10_holders", []))
                    }
                },
                # 合约权限分析
                "permission": {
                    "name": "Contract Safety",
                    "name_cn": "合约安全",
                    "description": "Contract permission and rug pull risk analysis",
                    "description_cn": "合约权限分析，检测Rug Pull风险",
                    "score": permission_result.get("score", 0),
                    "max_score": 30,
                    "weight": "30%",
                    "risk_level": permission_result.get("risk_level", "unknown"),
                    "metrics": {
                        "has_owner": permission_result.get("owner_info", {}).get("has_owner", False),
                        "owner_address": permission_result.get("owner_info", {}).get("owner_address"),
                        "is_renounced": permission_result.get("owner_info", {}).get("is_renounced", False),
                        "is_multisig": permission_result.get("owner_info", {}).get("is_multisig", False),
                        "is_proxy": permission_result.get("proxy_info", {}).get("is_proxy", False),
                        "dangerous_functions": [
                            {
                                "category": f["category"],
                                "signature": f["signature"]
                            }
                            for f in permission_result.get("dangerous_functions", {}).get("dangerous_functions", [])
                        ],
                        "risk_summary": permission_result.get("risk_summary", [])
                    }
                }
            }
        }

    def _format_top_holders(self, top_holders: List) -> List[Dict]:
        """格式化 Top 持有者数据供前端展示"""
        formatted = []
        for i, holder in enumerate(top_holders[:10]):
            if isinstance(holder, (list, tuple)) and len(holder) >= 3:
                formatted.append({
                    "rank": i + 1,
                    "address": holder[0],
                    "address_short": f"{holder[0][:6]}...{holder[0][-4:]}",
                    "balance": holder[1],
                    "percentage": round(holder[2], 2)
                })
        return formatted

    def _print_summary(self, result: Dict):
        """打印评分摘要"""
        print(f"\n{'='*60}")
        print(f"  SCORING SUMMARY")
        print(f"{'='*60}")

        overview = result["overview"]
        scores = result["scores"]

        print(f"\n  EOA Analysis:        {scores['eoa']['score']:>5.1f} / 40")
        print(f"  Holder Analysis:     {scores['holder']['score']:>5.1f} / 30")
        print(f"  Permission Analysis: {scores['permission']['score']:>5.1f} / 30")
        print(f"  " + "-"*40)
        print(f"  TOTAL SCORE:         {overview['total_score']:>5.1f} / 100")

        print(f"\n  Risk Level: {overview['risk_label']} ({overview['risk_label_cn']})")

        if result["risk_tags"]:
            print(f"\n  Risk Tags:")
            for tag in result["risk_tags"]:
                type_marker = {"success": "[OK]", "warning": "[!]", "danger": "[X]"}.get(tag["type"], "[ ]")
                print(f"    {type_marker} {tag['label']} ({tag['label_cn']})")

        print(f"\n{'='*60}\n")

    def to_json(self, result: Dict, indent: int = 2) -> str:
        """将结果转换为 JSON 字符串"""
        return json.dumps(result, indent=indent, ensure_ascii=False, default=str)

    def save_result(self, result: Dict, output_dir: str = "output") -> str:
        """
        保存评分结果到文件

        Args:
            result: 评分结果
            output_dir: 输出目录

        Returns:
            保存的文件路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名: {token_address}_{timestamp}.json
        token_short = result["token_address"][:10]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{token_short}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # 保存完整结果
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n  [OK] Result saved to: {filepath}")
        return filepath


def quick_score(
    token_address: str,
    network: str = "monad_testnet",
    mode: str = "auto",
    use_nansen: bool = True
) -> Dict:
    """
    快速评分函数 (便捷接口)

    Args:
        token_address: 代币地址
        network: 网络名称
        mode: 分析模式 ("auto", "fast", "deep")
        use_nansen: 是否使用 Nansen (默认 True)

    Returns:
        评分结果
    """
    client = Web3Client(network=network)

    nansen = None
    if use_nansen:
        try:
            nansen = NansenClient()
        except Exception as e:
            print(f"[!] Nansen initialization failed: {e}")
            print("[!] Falling back to deep mode...")

    scorer = TotalScorer(client, nansen=nansen)
    return scorer.score_token(token_address, mode=mode)


# 使用示例
if __name__ == "__main__":
    import time
    from dotenv import load_dotenv
    load_dotenv()

    # 测试代币地址 (WMON)
    test_token = os.getenv("TEST_TOKEN_ADDRESS", "0x3bd359C1119dA7Da1D913D1C4D2B7c461115433A")

    print("=" * 60)
    print("Token Scoring Test")
    print("=" * 60)
    print(f"\nToken: {test_token}")

    # ========== 快速模式 (Nansen) ==========
    print("\n" + "=" * 60)
    print("Mode 1: FAST (Nansen API)")
    print("=" * 60)

    try:
        # 创建 Nansen 客户端
        nansen = NansenClient()

        # 尝试创建 Web3 客户端 (可选，权限分析需要)
        try:
            client = Web3Client(network="monad_mainnet")
        except Exception as e:
            print(f"  [!] Web3 client failed: {e}")
            print(f"  [i] Running without Web3 (permission analysis will be skipped)...")
            client = None

        # 创建综合评分器 (带 Nansen)
        scorer = TotalScorer(client, nansen=nansen)

        # 执行快速评分 (limit=500 获取更多数据)
        result = scorer.score_token(
            token_address=test_token,
            mode="fast",
            time_window_hours=1,
            limit=500
        )

        print(f"\n=== Data Sources ===")
        print(f"EOA: {result.get('data_sources', {}).get('eoa', 'unknown')}")
        print(f"Holder: {result.get('data_sources', {}).get('holder', 'unknown')}")
        print(f"Permission: {result.get('data_sources', {}).get('permission', 'unknown')}")

        # 保存结果
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
        scorer.save_result(result, output_dir)

    except Exception as e:
        print(f"\n[!] Fast mode failed: {e}")
        import traceback
        traceback.print_exc()

    # ========== 深度模式 (BlockPi) - 可选 ==========
    # 注意: 深度模式会调用大量 RPC，耗时较长
    # 取消下面的注释来测试深度模式

    # print("\n" + "=" * 60)
    # print("Mode 2: DEEP (BlockPi RPC)")
    # print("=" * 60)
    #
    # try:
    #     client = Web3Client(network="monad_mainnet")
    #     scorer = TotalScorer(client)  # 不传 Nansen
    #
    #     result = scorer.score_token(
    #         token_address=test_token,
    #         mode="deep",
    #         time_window_hours=1
    #     )
    #
    #     print(f"\n=== Data Sources ===")
    #     print(f"EOA: {result.get('data_sources', {}).get('eoa', 'unknown')}")
    #     print(f"Holder: {result.get('data_sources', {}).get('holder', 'unknown')}")
    #
    # except Exception as e:
    #     print(f"\n[!] Deep mode failed: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
