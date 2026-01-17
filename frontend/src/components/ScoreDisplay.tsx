import type { ScoreData, TopHolder } from '../App'

interface Props {
  data: ScoreData
}

// 根据分数获取状态配置
function getScoreStatus(score: number) {
  if (score >= 80) return { label: 'Fully Protected', labelCn: '完全防护', color: '#22c55e', status: 'safe' }
  if (score >= 60) return { label: 'Moderate Risk', labelCn: '中等风险', color: '#eab308', status: 'warning' }
  if (score >= 40) return { label: 'High Risk', labelCn: '高风险', color: '#f97316', status: 'danger' }
  return { label: 'Extreme Risk', labelCn: '极高风险', color: '#ef4444', status: 'critical' }
}

// 圆形进度条组件
function CircularProgress({ score, size = 180, strokeWidth = 8 }: { score: number; size?: number; strokeWidth?: number }) {
  const status = getScoreStatus(score)
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const offset = circumference - (score / 100) * circumference

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        {/* 背景圆环 */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#1f1f2e"
          strokeWidth={strokeWidth}
          fill="none"
        />
        {/* 进度圆环 */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={status.color}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-1000 ease-out"
          style={{
            filter: `drop-shadow(0 0 8px ${status.color}40)`
          }}
        />
      </svg>
      {/* 中心内容 */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <svg className="w-6 h-6 mb-1" fill={status.color} viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
        </svg>
        <span className="text-4xl font-bold text-white">{Math.round(score)}</span>
        <span className="text-sm text-gray-500">/100</span>
      </div>
    </div>
  )
}

// 用户类型分布条
function UserTypeBar({ label, labelCn, percentage, color }: { label: string; labelCn: string; percentage: number; color: string }) {
  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center gap-2">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-600 text-sm">{labelCn}</span>
      </div>
      <div className="flex items-center gap-3">
        <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${Math.min(percentage, 100)}%`, backgroundColor: color }}
          />
        </div>
        <span className="text-white font-medium w-14 text-right">{percentage.toFixed(1)}%</span>
      </div>
    </div>
  )
}

export default function ScoreDisplay({ data }: Props) {
  const { overview, scores } = data
  const status = getScoreStatus(overview.total_score)

  // 从后端获取真实数据
  const eoaMetrics = scores.eoa.metrics as any
  const holderMetrics = scores.holder.metrics

  // 真实数据
  const totalAddresses = eoaMetrics.total_addresses || 0

  // 从 Nansen API 获取的用户类型分布
  const userTypes = eoaMetrics.user_types || {}
  const realUsersPercentage = userTypes.real_users?.percentage || eoaMetrics.eoa_percentage || 0
  const smartMoneyPercentage = userTypes.smart_money?.percentage || 0
  const dexPoolPercentage = userTypes.dex_pool?.percentage || 0
  const botsPercentage = userTypes.bots?.percentage || 0

  // Bot 活动分析 - 使用机器人占比
  const botVolumeShare = botsPercentage + dexPoolPercentage
  const botStatus = botVolumeShare < 20 ? 'Organic' : botVolumeShare < 50 ? 'Moderate' : 'High Bot Activity'
  const botStatusCn = botVolumeShare < 20 ? '自然' : botVolumeShare < 50 ? '中等' : '高机器人活动'

  // Top10 持有者集中度
  const top10Percentage = holderMetrics.top10_percentage || 0

  return (
    <div className="space-y-6">
      {/* 主评分卡片 */}
      <div className="bg-[#12121a] rounded-2xl border border-white/5 p-8">
        <div className="flex items-center gap-8">
          {/* 圆形评分 */}
          <CircularProgress score={overview.total_score} />

          {/* 评分信息 */}
          <div className="flex-1">
            <p className="text-gray-500 text-sm mb-1">SHIELD SCORE  护盾评分</p>
            <h2 className="text-3xl font-bold mb-1" style={{ color: status.color }}>
              {status.label}  {status.labelCn}
            </h2>
            <p className="text-gray-400 text-sm mb-4">
              {overview.total_score >= 80
                ? 'Highly organic activity with healthy distribution'
                : overview.total_score >= 60
                ? 'Moderate risk detected in token distribution'
                : 'High concentration detected - proceed with caution'
              }
            </p>
            <p className="text-gray-500 text-sm">
              {overview.total_score >= 80
                ? '高度有机活动，健康分布'
                : overview.total_score >= 60
                ? '代币分布存在中等风险'
                : '检测到高集中度 - 请谨慎操作'
              }
            </p>

            {/* 进度条 */}
            <div className="mt-4 h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-1000"
                style={{
                  width: `${overview.total_score}%`,
                  backgroundColor: status.color
                }}
              />
            </div>

            {/* Token 地址 */}
            <div className="mt-4 flex items-center gap-2">
              <span className="text-gray-500 text-sm">Token: 代币:</span>
              <a
                href={`https://monad.socialscan.io/token/${data.token_address}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-monad-purple hover:underline font-mono text-sm"
              >
                {data.token_address.slice(0, 10)}...{data.token_address.slice(-8)}
              </a>
              <button
                onClick={() => navigator.clipboard.writeText(data.token_address)}
                className="text-gray-500 hover:text-white transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 三列卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Concentration 卡片 */}
        <div className="bg-[#12121a] rounded-2xl border border-white/5 p-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-lg">📊</span>
            <h3 className="text-white font-semibold">Concentration  专注度</h3>
          </div>

          <p className="text-gray-500 text-sm mb-2">Top 10% Volume Share  前 10%市场份额</p>
          <p className="text-4xl font-bold text-white mb-4">
            {top10Percentage.toFixed(1)}%
          </p>

          <div className="h-1 bg-gray-800 rounded-full overflow-hidden mb-4">
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.min(top10Percentage, 100)}%`,
                backgroundColor: top10Percentage > 80 ? '#ef4444' : top10Percentage > 50 ? '#f97316' : '#22c55e'
              }}
            />
          </div>

          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${top10Percentage > 80 ? 'bg-red-500' : top10Percentage > 50 ? 'bg-orange-500' : 'bg-green-500'}`}></span>
            <span className={top10Percentage > 80 ? 'text-red-400' : top10Percentage > 50 ? 'text-orange-400' : 'text-green-400'}>
              {top10Percentage > 80 ? 'Highly Concentrated  高度集中' : top10Percentage > 50 ? 'Moderately Concentrated  中度集中' : 'Well Distributed  分布良好'}
            </span>
          </div>
          <p className="text-gray-600 text-xs mt-1">
            {top10Percentage > 80 ? 'High concentration - few addresses dominate' : top10Percentage > 50 ? 'Moderate concentration detected' : 'Healthy distribution across holders'}
          </p>
          <p className="text-gray-700 text-xs">
            {top10Percentage > 80 ? '高浓度 - 少数地址主导' : top10Percentage > 50 ? '检测到中等浓度' : '持有者分布健康'}
          </p>
        </div>

        {/* User Types 卡片 */}
        <div className="bg-[#12121a] rounded-2xl border border-white/5 p-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-lg">👥</span>
            <h3 className="text-white font-semibold">User Types  用户类型</h3>
          </div>

          <div className="space-y-1">
            <UserTypeBar label="Real Users" labelCn="真实用户" percentage={realUsersPercentage} color="#22c55e" />
            <UserTypeBar label="Smart Money" labelCn="智能资金" percentage={smartMoneyPercentage} color="#eab308" />
            <UserTypeBar label="DEX/Pool" labelCn="DEX/池" percentage={dexPoolPercentage} color="#3b82f6" />
            <UserTypeBar label="Bots" labelCn="机器人" percentage={botsPercentage} color="#f97316" />
          </div>
        </div>

        {/* Bot Activity 卡片 */}
        <div className="bg-[#12121a] rounded-2xl border border-white/5 p-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-lg">🤖</span>
            <h3 className="text-white font-semibold">Bot Activity  机器人活动</h3>
          </div>

          <p className="text-gray-500 text-sm mb-2">Bot Volume Share  机器人容量份额</p>
          <p className="text-4xl font-bold text-white mb-4">{botVolumeShare.toFixed(1)}%</p>

          <div className="h-1 bg-gray-800 rounded-full overflow-hidden mb-4">
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.min(botVolumeShare, 100)}%`,
                backgroundColor: botVolumeShare < 20 ? '#22c55e' : botVolumeShare < 50 ? '#eab308' : '#ef4444'
              }}
            />
          </div>

          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${botVolumeShare < 20 ? 'bg-green-500' : botVolumeShare < 50 ? 'bg-yellow-500' : 'bg-red-500'}`}></span>
            <span className={botVolumeShare < 20 ? 'text-green-400' : botVolumeShare < 50 ? 'text-yellow-400' : 'text-red-400'}>
              {botStatus}  {botStatusCn}
            </span>
          </div>
          <p className="text-gray-600 text-xs mt-1">
            {botVolumeShare < 20 ? 'Minimal bot activity, mostly real users' : botVolumeShare < 50 ? 'Some bot activity detected' : 'High bot activity detected'}
          </p>
          <p className="text-gray-700 text-xs">
            {botVolumeShare < 20 ? '极少数机器人活动，主要是真实用户' : botVolumeShare < 50 ? '检测到一些机器人活动' : '检测到高机器人活动'}
          </p>
        </div>
      </div>

      {/* Top Interactors 表格 */}
      {holderMetrics.top10_holders && holderMetrics.top10_holders.length > 0 && (
        <div className="bg-[#12121a] rounded-2xl border border-white/5 p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-lg">📋</span>
              <h3 className="text-white font-semibold">Top Interactors  主要互动者</h3>
            </div>
            <span className="text-gray-500 text-sm">
              {totalAddresses.toLocaleString()} addresses analyzed  分析了 {totalAddresses.toLocaleString()} 个地址
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-500 text-sm border-b border-white/5">
                  <th className="pb-3 font-medium">#</th>
                  <th className="pb-3 font-medium">Address  地址</th>
                  <th className="pb-3 font-medium">Type  类型</th>
                  <th className="pb-3 font-medium text-right">Volume  交易量</th>
                  <th className="pb-3 font-medium text-right">Share  分享</th>
                </tr>
              </thead>
              <tbody>
                {holderMetrics.top10_holders.slice(0, 10).map((holder: TopHolder, idx: number) => (
                  <tr key={holder.rank || idx} className="border-b border-white/5 hover:bg-white/[0.02] transition-colors">
                    <td className="py-3 text-gray-400">{holder.rank || idx + 1}</td>
                    <td className="py-3">
                      <a
                        href={`https://monad.socialscan.io/address/${holder.address}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-monad-purple hover:underline font-mono text-sm"
                      >
                        {holder.address_short || `${holder.address.slice(0, 6)}...${holder.address.slice(-4)}`}
                      </a>
                    </td>
                    <td className="py-3">
                      <span className="px-2 py-0.5 rounded text-xs bg-gray-800 text-gray-400">
                        EOA
                      </span>
                    </td>
                    <td className="py-3 text-right text-gray-300 font-mono text-sm">
                      {holder.balance?.toLocaleString(undefined, { maximumFractionDigits: 2 }) || '-'}
                    </td>
                    <td className="py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-16 h-1 bg-gray-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-monad-purple rounded-full"
                            style={{ width: `${Math.min(holder.percentage || 0, 100)}%` }}
                          />
                        </div>
                        <span className="text-gray-300 text-sm w-14 text-right">
                          {holder.percentage?.toFixed(2) || '0'}%
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}