# -*- coding: utf-8 -*-
"""
Nansen Token God Mode API 客户端
提供 Monad 链上代币持有者数据查询能力

优势:
- 原生支持 Monad 链
- 提供 Smart Money 标签 (DEX Bot, Bonding Curve 等)
- 一次 API 调用返回丰富的持有者数据
- 包含余额变化数据 (24h/7d/30d)
- 包含 USD 价值和持有占比

API 文档: https://docs.nansen.ai/api/token-god-mode/holders
"""

import os
import time
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv


class NansenError(Exception):
    """Nansen API 错误基类"""
    pass


class NansenAPIError(NansenError):
    """API 返回错误"""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Nansen API Error [{code}]: {message}")


class NansenNetworkError(NansenError):
    """网络请求错误"""
    pass


class NansenRateLimitError(NansenError):
    """API 限流错误"""
    pass


class SimpleCache:
    """简单的内存缓存，带 TTL"""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple] = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expire_time = self._cache[key]
            if time.time() < expire_time:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = (value, time.time() + self.ttl)

    def clear(self) -> None:
        self._cache.clear()


@dataclass
class NansenHolder:
    """Nansen 代币持有者信息"""
    address: str
    balance: float              # 持有数量
    balance_formatted: float    # 格式化余额 (兼容 Blockvision)
    percentage: float           # 占总供应量百分比
    rank: int = 0               # 排名
    is_contract: bool = False   # 是否是合约 (通过标签判断)

    # Nansen 特有字段
    address_label: str = ""     # 地址标签
    value_usd: float = 0        # USD 价值
    total_inflow: float = 0     # 总流入
    total_outflow: float = 0    # 总流出
    balance_change_24h: float = 0
    balance_change_7d: float = 0
    balance_change_30d: float = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def is_eoa(self) -> bool:
        """是否是 EOA (外部账户)"""
        return not self.is_contract

    @property
    def is_smart_money(self) -> bool:
        """是否是聪明钱/机器人"""
        if not self.address_label:
            return False
        label_lower = self.address_label.lower()
        return any(x in label_lower for x in ['smart money', 'bot', 'trading'])

    @property
    def is_dex_bot(self) -> bool:
        """是否是 DEX 交易机器人"""
        if not self.address_label:
            return False
        return 'dex' in self.address_label.lower() and 'bot' in self.address_label.lower()


class NansenClient:
    """
    Nansen Token God Mode API 客户端

    用于获取 Monad 链上代币的持有者数据，替代 Blockvision。

    使用示例:
        >>> client = NansenClient()
        >>> result = client.get_token_holders("0x...", page_size=100)
        >>> for h in result["holders"]:
        ...     print(f"{h.address}: {h.percentage}% ({h.address_label})")
    """

    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 1

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        auto_retry: bool = True,
        cache_ttl: int = 300
    ):
        """
        初始化 Nansen 客户端

        Args:
            api_key: API Key，不提供则从环境变量 NANSEN_API_KEY 读取
            base_url: API 基础 URL
            timeout: 请求超时时间 (秒)
            auto_retry: 是否自动重试
            cache_ttl: 缓存有效期 (秒)
        """
        load_dotenv()

        self.api_key = api_key or os.getenv('NANSEN_API_KEY', '')
        self.base_url = base_url or os.getenv('NANSEN_BASE_URL', 'https://api.nansen.ai/api/v1')

        if not self.api_key:
            raise ValueError(
                "Nansen API Key not found. "
                "Please set NANSEN_API_KEY in .env or pass api_key parameter."
            )

        self.timeout = timeout
        self.auto_retry = auto_retry

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "*/*",
            "apiKey": self.api_key
        })

        self._request_count = 0
        self._last_request_time = 0
        self._cache = SimpleCache(ttl_seconds=cache_ttl)

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def get_token_holders(
        self,
        contract_address: str,
        page_index: int = 1,
        page_size: int = 100,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        获取代币持有者列表 (支持分页)

        Args:
            contract_address: 代币合约地址
            page_index: 页码 (从 1 开始)
            page_size: 每页数量 (最大 100)
            use_cache: 是否使用缓存

        Returns:
            {
                "total": 当前页返回数量,
                "holders": [NansenHolder, ...],
                "page_index": 当前页码,
                "page_size": 每页数量,
                "is_last_page": 是否是最后一页
            }
        """
        cache_key = f"nansen_holders_{contract_address.lower()}_{page_index}_{page_size}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Nansen API 分页需要放在 pagination 对象中
        payload = {
            "chain": "monad",
            "token_address": self._normalize_address(contract_address),
            "aggregate_by_entity": False,
            "pagination": {
                "page": page_index,
                "per_page": min(page_size, 100)  # 最大100
            }
        }

        result = self._request("POST", "tgm/holders", data=payload)

        holders = []
        data_list = result.get("data", []) if isinstance(result, dict) else []
        pagination_info = result.get("pagination", {})

        # 获取总供应量 (用于计算占比)
        total_supply = float(result.get("total_supply", 0)) or float(result.get("totalSupply", 0))

        for idx, item in enumerate(data_list):
            amount = float(item.get("token_amount", 0))
            # 尝试多种可能的字段名获取占比
            raw_percentage = (
                item.get("ownership_percentage") or
                item.get("percentage") or
                item.get("share") or
                item.get("pct") or
                0
            )

            # 如果 API 返回了有效的占比
            if raw_percentage and float(raw_percentage) > 0:
                percentage = float(raw_percentage) * 100 if float(raw_percentage) <= 1 else float(raw_percentage)
            elif total_supply > 0 and amount > 0:
                # 从总供应量计算占比
                percentage = (amount / total_supply) * 100
            else:
                # 无法计算占比时，标记为 -1 (后续处理)
                percentage = -1

            label = item.get("address_label", "")

            # 通过标签判断是否是合约
            is_contract = any(x in label.lower() for x in [
                'contract', 'dex', 'cex', 'exchange', 'pool', 'vault',
                'bonding_curve', 'router', 'factory'
            ]) if label else False

            holders.append(NansenHolder(
                address=item.get("address", ""),
                balance=int(amount),
                balance_formatted=amount,
                percentage=percentage,
                rank=(page_index - 1) * page_size + idx + 1,  # 全局排名
                is_contract=is_contract,
                address_label=label,
                value_usd=float(item.get("value_usd", 0)),
                total_inflow=float(item.get("total_inflow", 0)),
                total_outflow=float(item.get("total_outflow", 0)),
                balance_change_24h=float(item.get("balance_change_24h", 0)),
                balance_change_7d=float(item.get("balance_change_7d", 0)),
                balance_change_30d=float(item.get("balance_change_30d", 0))
            ))

        # 按持有量排序
        holders.sort(key=lambda x: x.balance_formatted, reverse=True)

        response = {
            "total": len(data_list),
            "holders": holders,
            "page_index": page_index,
            "page_size": page_size,
            "is_last_page": pagination_info.get("is_last_page", True)
        }

        if use_cache:
            self._cache.set(cache_key, response)

        return response

    def get_all_holders(
        self,
        contract_address: str,
        max_pages: int = 100,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        获取所有持有者 (自动分页)

        Args:
            contract_address: 代币合约地址
            max_pages: 最大页数限制 (防止无限循环)
            use_cache: 是否使用缓存

        Returns:
            {
                "total": 总持有者数量,
                "holders": [NansenHolder, ...],
                "pages_fetched": 获取的页数
            }
        """
        cache_key = f"nansen_all_holders_{contract_address.lower()}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        all_holders = []
        page = 1
        per_page = 100

        while page <= max_pages:
            result = self.get_token_holders(
                contract_address,
                page_index=page,
                page_size=per_page,
                use_cache=False  # 不缓存单页结果
            )

            all_holders.extend(result["holders"])

            if result.get("is_last_page", True):
                break

            page += 1

        # 按持有量排序并重新计算排名
        all_holders.sort(key=lambda x: x.balance_formatted, reverse=True)
        for idx, h in enumerate(all_holders):
            h.rank = idx + 1

        response = {
            "total": len(all_holders),
            "holders": all_holders,
            "pages_fetched": page
        }

        if use_cache:
            self._cache.set(cache_key, response)

        return response

    def get_top_holders(
        self,
        contract_address: str,
        top_n: int = 10
    ) -> List[NansenHolder]:
        """获取 Top N 持有者"""
        result = self.get_token_holders(contract_address, page_size=min(top_n, 100))
        return result["holders"][:top_n]

    def get_top_holders_percentage(
        self,
        contract_address: str,
        top_n: int = 10
    ) -> float:
        """获取 Top N 持有者的总占比"""
        holders = self.get_top_holders(contract_address, top_n)
        return sum(h.percentage for h in holders)

    def get_holder_count(self, contract_address: str) -> int:
        """获取代币总持有者数量"""
        result = self.get_token_holders(contract_address, page_size=1)
        return result["total"]

    def count_unique_eoa(
        self,
        contract_address: str,
        limit: int = 1000,
        fetch_all: bool = False
    ) -> Dict[str, Any]:
        """
        统计代币的独立 EOA 数量和用户类型分布 (用于 EOA 活跃度评分)

        Args:
            contract_address: 代币合约地址
            limit: 分析的持有者数量上限 (仅当 fetch_all=False 时生效)
            fetch_all: 是否获取所有持有者

        Returns:
            {
                "unique_eoa_count": 独立 EOA 数量,
                "total_addresses": 总地址数量,
                "eoa_ratio": EOA 占比 (0-100),
                "smart_money_count": 聪明钱数量,
                "dex_pool_count": DEX/Pool 数量,
                "bot_count": 机器人数量,
                "holders_analyzed": 分析的持有者数量,
                "user_types": {
                    "real_users": { "count": x, "percentage": y },
                    "smart_money": { "count": x, "percentage": y },
                    "dex_pool": { "count": x, "percentage": y },
                    "bots": { "count": x, "percentage": y }
                }
            }
        """
        if fetch_all:
            result = self.get_all_holders(contract_address)
            holders = result["holders"]
        else:
            # 分页获取直到 limit
            holders = []
            page = 1
            per_page = 100
            while len(holders) < limit:
                result = self.get_token_holders(
                    contract_address,
                    page_index=page,
                    page_size=per_page,
                    use_cache=True
                )
                holders.extend(result["holders"])
                if result.get("is_last_page", True):
                    break
                page += 1
            holders = holders[:limit]

        total = len(holders)
        if total == 0:
            return {
                "unique_eoa_count": 0,
                "total_addresses": 0,
                "eoa_ratio": 0,
                "smart_money_count": 0,
                "dex_pool_count": 0,
                "bot_count": 0,
                "holders_analyzed": 0,
                "user_types": {
                    "real_users": {"count": 0, "percentage": 0},
                    "smart_money": {"count": 0, "percentage": 0},
                    "dex_pool": {"count": 0, "percentage": 0},
                    "bots": {"count": 0, "percentage": 0}
                }
            }

        # 分类统计
        eoa_count = 0
        smart_money_count = 0
        dex_pool_count = 0
        bot_count = 0

        for h in holders:
            label = h.address_label.lower() if h.address_label else ""
            label_raw = h.address_label if h.address_label else ""

            # 检查是否包含机器人 emoji 🤖
            has_bot_emoji = '🤖' in label_raw
            # 检查是否包含交易所 emoji 🏦
            has_cex_emoji = '🏦' in label_raw

            # DEX/Pool 判断 (优先级最高)
            if any(x in label for x in ['pool', 'liquidity', 'amm', 'uniswap', 'pancakeswap', 'kuru', 'curve', '3pool']):
                dex_pool_count += 1
            # CEX/交易所 判断 - 归类到 Smart Money
            elif has_cex_emoji or any(x in label for x in ['bybit', 'bitget', 'coinbase', 'binance', 'okx', 'exchange', 'hot wallet']):
                smart_money_count += 1
            # Bot 判断 (包含 �� emoji 或关键词，但不是 pool)
            elif has_bot_emoji or any(x in label for x in ['bot', 'mev', 'arbitrage', 'sniper', 'trading bot', 'gearbox', 'morpho', 'neverland', 'curvance', 'manifold']):
                bot_count += 1
            # Smart Money 判断
            elif any(x in label for x in ['smart money', 'whale', 'millionaire', 'high balance', 'fund']):
                smart_money_count += 1
            # EOA (真实用户) - 没有特殊标签的地址
            else:
                eoa_count += 1

        # 计算百分比
        user_types = {
            "real_users": {
                "count": eoa_count,
                "percentage": round(eoa_count / total * 100, 2)
            },
            "smart_money": {
                "count": smart_money_count,
                "percentage": round(smart_money_count / total * 100, 2)
            },
            "dex_pool": {
                "count": dex_pool_count,
                "percentage": round(dex_pool_count / total * 100, 2)
            },
            "bots": {
                "count": bot_count,
                "percentage": round(bot_count / total * 100, 2)
            }
        }

        return {
            "unique_eoa_count": eoa_count,
            "total_addresses": total,
            "eoa_ratio": round(eoa_count / total * 100, 2),
            "smart_money_count": smart_money_count,
            "dex_pool_count": dex_pool_count,
            "bot_count": bot_count,
            "holders_analyzed": total,
            "user_types": user_types
        }

    def is_available(self) -> bool:
        """检查 API 是否可用"""
        try:
            self.get_token_holders(
                "0x0000000000000000000000000000000000000000",
                page_size=1
            )
            return True
        except NansenRateLimitError:
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        return {
            "request_count": self._request_count,
            "api_key_prefix": self.api_key[:8] + "..." if len(self.api_key) > 8 else self.api_key,
            "base_url": self.base_url
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict:
        """发送 API 请求"""
        url = f"{self.base_url}/{endpoint}"

        retries = 0
        last_error = None

        while retries <= (self.MAX_RETRIES if self.auto_retry else 0):
            try:
                self._request_count += 1
                self._last_request_time = time.time()

                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=self.timeout)
                else:
                    response = self.session.post(url, params=params, json=data, timeout=self.timeout)

                if response.status_code == 429:
                    raise NansenRateLimitError("API rate limit exceeded")

                if response.status_code == 401:
                    raise NansenAPIError(401, "Unauthorized - check your API key")

                if response.status_code == 403:
                    raise NansenAPIError(403, "Forbidden - API key may not have access")

                response.raise_for_status()
                result = response.json()

                if isinstance(result, dict) and result.get("error"):
                    raise NansenAPIError(result.get("code", -1), result.get("error", "Unknown error"))

                return result

            except NansenRateLimitError:
                raise
            except NansenAPIError:
                raise
            except requests.exceptions.Timeout as e:
                last_error = NansenNetworkError(f"Request timeout: {e}")
            except requests.exceptions.RequestException as e:
                last_error = NansenNetworkError(f"Request failed: {e}")
            except Exception as e:
                last_error = NansenError(f"Unexpected error: {e}")

            retries += 1
            if retries <= self.MAX_RETRIES and self.auto_retry:
                time.sleep(self.RETRY_DELAY * retries)

        raise last_error

    def _normalize_address(self, address: str) -> str:
        """标准化地址格式"""
        address = address.strip().lower()
        if not address.startswith("0x"):
            address = "0x" + address
        return address

    def __repr__(self) -> str:
        return f"NansenClient(api_key={self.api_key[:8]}..., requests={self._request_count})"


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    load_dotenv()

    print("=" * 60)
    print("Nansen Client Test (With Pagination)")
    print("=" * 60)

    try:
        client = NansenClient()
        print(f"\nClient: {client}")

        test_token = os.getenv("TEST_TOKEN_ADDRESS", "0x3bd359C1119dA7Da1D913D1C4D2B7c461115433A")
        print(f"\nTest Token: {test_token}")

        # 测试分页获取
        print("\n--- Pagination Test ---")
        print("Fetching page 1 (100 holders)...")
        result1 = client.get_token_holders(test_token, page_index=1, page_size=100)
        print(f"  Page 1: {len(result1['holders'])} holders, is_last_page: {result1.get('is_last_page')}")

        if not result1.get("is_last_page"):
            print("Fetching page 2...")
            result2 = client.get_token_holders(test_token, page_index=2, page_size=100)
            print(f"  Page 2: {len(result2['holders'])} holders, is_last_page: {result2.get('is_last_page')}")

        # 显示 Top 5
        print("\n--- Top 5 Holders ---")
        for h in result1["holders"][:5]:
            eoa_tag = "EOA" if h.is_eoa else "Contract"
            label = h.address_label.encode('ascii', 'ignore').decode('ascii') if h.address_label else ""
            label_str = f" [{label}]" if label else ""
            print(f"  #{h.rank} {h.address[:16]}... : {h.percentage:.4f}% ({eoa_tag}){label_str}")

        print(f"\nTop 10 Percentage: {client.get_top_holders_percentage(test_token):.4f}%")

        # 测试获取更多持有者
        print("\n--- Fetch 500 Holders ---")
        eoa_stats = client.count_unique_eoa(test_token, limit=500)
        print(f"Holders Analyzed: {eoa_stats['holders_analyzed']}")
        print(f"Unique EOA: {eoa_stats['unique_eoa_count']}")
        print(f"EOA Ratio: {eoa_stats['eoa_ratio']}%")
        print(f"Smart Money Count: {eoa_stats['smart_money_count']}")

        print("\n--- Client Stats ---")
        stats = client.get_stats()
        print(f"Total Requests: {stats['request_count']}")

        print("\n" + "=" * 60)
        print("All tests passed!")

    except NansenError as e:
        print(f"\nNansen Error: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
