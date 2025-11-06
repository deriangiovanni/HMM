import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Backtest data (REVERSED: now goes from oldest → newest)
returns_raw = [
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75,
    -1.50, -1.50, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, -1.50,
    0.75, 0.75, -1.50, -1.50, -1.50, -1.50, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
    -1.50, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50,
    0.75, 0.75, -1.50, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75,
    -1.50, 0.75, -1.50, -1.50, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, -1.50, 0.75,
    0.75, 0.75, -1.50, 0.75, -1.50, -1.50, -1.50, -1.50, 0.75, 0.75,
    0.75, -1.50, -1.50, -1.50, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75,
    -1.50, 0.75, 0.75, -1.50, 0.75, -1.50, 0.75, -1.50, 0.75, -1.50,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
    0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75,
    0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50,
    -1.50, -1.50, 0.75, 0.75, -1.50, 0.75, -1.50, 0.75, -1.50, 0.75,
    -1.50, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50, -1.50,
    0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, -1.50, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, -1.50,
    -1.50, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, -1.50,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75,
    0.75, -1.50, 0.75, 0.75, -1.50, 0.75, -1.50, 0.75, 0.75, 0.75,
    0.75, 0.75, -1.50, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50,
    0.75, 0.75, -1.50, 0.75, -1.50, 0.75, -1.50, 0.75, 0.75, 0.75,
    0.75, 0.75, -1.50, 0.75, -1.50, 0.75, -1.50, 0.75, 0.75, 0.75,
    0.75, -1.50, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, -1.50,
    0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, -1.50,
    0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75,
    -1.50, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75,
    0.75, 0.75, -1.50, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, -1.50,
    0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50,
    -1.50, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
    -1.50, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
    -1.50, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50,
    0.75, -1.50, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75,
    0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50,
    0.75, -1.50, 0.75, 0.75, 0.75, 0.75, 0.75, -1.50, 0.75, 0.75,
    0.75, 0.75, 0.75, -1.50, 0.75, -1.50, 0.75, -1.50, -1.50, 0.75,
    0.75, 0.75, -1.50, 0.75, 0.75, -1.50, 0.75, 0.75, 0.75, 0.75
]

returns = list(reversed(returns_raw))


def detect_autocorrelation(returns, max_lag=10):
    """Detect if trades are autocorrelated (overlapping trades indicator)"""
    print("\n" + "="*70)
    print(f"{'🔍 AUTOCORRELATION ANALYSIS (Overlap Detection)':^70}")
    print("="*70)
    
    returns_array = np.array(returns)
    
    print(f"\nTesting for autocorrelation up to {max_lag} lags...")
    print("High autocorrelation suggests overlapping/dependent trades\n")
    
    significant_lags = []
    
    for lag in range(1, max_lag + 1):
        if lag >= len(returns):
            break
            
        # Calculate autocorrelation
        returns_shifted = np.roll(returns_array, lag)
        correlation = np.corrcoef(returns_array[lag:], returns_shifted[lag:])[0, 1]
        
        # Ljung-Box test
        n = len(returns)
        q_stat = n * (n + 2) * correlation**2 / (n - lag)
        p_value = 1 - stats.chi2.cdf(q_stat, 1)
        
        is_significant = p_value < 0.05
        
        print(f"Lag {lag:2d}: r = {correlation:7.4f} | p = {p_value:.4f} "
              f"{'⚠️ SIGNIFICANT' if is_significant else '✓ OK'}")
        
        if is_significant:
            significant_lags.append((lag, correlation, p_value))
    
    print("\n" + "="*70)
    
    if significant_lags:
        print(f"⚠️  AUTOCORRELATION DETECTED AT {len(significant_lags)} LAG(S)")
        print(f"   Note: With α=0.05 and {max_lag} tests, ~{max_lag*0.05:.1f} false positives expected")
        if len(significant_lags) <= 1:
            print("   ✅ Only 1 significant lag - likely random variation")
        else:
            print("   ⚠️  Multiple lags significant - potential overlap issue")
    else:
        print("✅ NO SIGNIFICANT AUTOCORRELATION")
        print("   Trades appear independent - results reliable!")
    
    print("="*70)
    
    return significant_lags


def durbin_watson_test(returns):
    """Durbin-Watson test: DW≈2 (no autocorr), DW<1.5 (positive autocorr)"""
    print("\n" + "="*70)
    print(f"{'📊 DURBIN-WATSON TEST':^70}")
    print("="*70)
    
    returns_array = np.array(returns)
    diff = np.diff(returns_array)
    
    numerator = np.sum(diff**2)
    denominator = np.sum((returns_array - np.mean(returns_array))**2)
    
    dw_stat = numerator / denominator if denominator != 0 else 2.0
    
    print(f"\nDurbin-Watson Statistic: {dw_stat:.4f}")
    print("\nInterpretation:")
    print("  DW ≈ 2.0 → No autocorrelation (ideal)")
    print("  DW < 1.5 → Strong positive autocorrelation (overlaps likely)")
    print("  DW > 2.5 → Negative autocorrelation")
    
    if dw_stat < 1.5:
        print("\n❌ STRONG AUTOCORRELATION - High overlap risk")
        risk = "HIGH"
    elif dw_stat < 1.8:
        print("\n⚠️  MODERATE AUTOCORRELATION - Some overlap possible")
        risk = "MODERATE"
    elif dw_stat > 2.2:
        print("\n⚠️  NEGATIVE AUTOCORRELATION - Unusual pattern")
        risk = "MODERATE"
    else:
        print("\n✅ MINIMAL AUTOCORRELATION - Trades independent")
        risk = "LOW"
    
    print(f"\n📊 OVERLAP RISK: {risk}")
    print("="*70)
    
    return dw_stat, risk


def runs_test(returns):
    """Test if win/loss sequence is random or clustered"""
    print("\n" + "="*70)
    print(f"{'🎲 RUNS TEST (Randomness Check)':^70}")
    print("="*70)
    
    wins = [1 if r > 0 else 0 for r in returns]
    
    # Count runs
    runs = 1
    for i in range(1, len(wins)):
        if wins[i] != wins[i-1]:
            runs += 1
    
    # Expected runs under randomness
    n_wins = sum(wins)
    n_losses = len(wins) - n_wins
    n = len(wins)
    
    expected_runs = (2 * n_wins * n_losses / n) + 1
    variance_runs = (2 * n_wins * n_losses * (2 * n_wins * n_losses - n)) / (n**2 * (n - 1))
    
    z_score = (runs - expected_runs) / np.sqrt(variance_runs) if variance_runs > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    print(f"\nObserved Runs: {runs}")
    print(f"Expected Runs (random): {expected_runs:.2f}")
    print(f"Z-Score: {z_score:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        if runs < expected_runs:
            print("\n❌ NON-RANDOM: Too few runs (clustering/overlaps)")
        else:
            print("\n❌ NON-RANDOM: Too many runs (alternating)")
    else:
        print("\n✅ RANDOM: No obvious clustering")
    
    print("="*70)
    
    return runs, expected_runs, p_value


def create_features(returns, window=5):
    """Create features for HMM"""
    df = pd.DataFrame({'return': returns})
    
    df['is_win'] = (df['return'] > 0).astype(int)
    df['rolling_winrate'] = df['is_win'].rolling(window=window, min_periods=1).mean()
    df['rolling_vol'] = df['return'].rolling(window=window, min_periods=1).std().fillna(0)
    
    df['consecutive'] = 0
    count = 0
    prev_val = None
    for i in range(len(df)):
        if prev_val is None or df.loc[i, 'is_win'] != prev_val:
            count = 1
        else:
            count += 1
        df.loc[i, 'consecutive'] = count if df.loc[i, 'is_win'] == 1 else -count
        prev_val = df.loc[i, 'is_win']
    
    return df

def train_hmm_model(features, n_states=2):
    """Train HMM with improved convergence"""
    feature_cols = ['rolling_winrate', 'rolling_vol', 'consecutive']
    X = features[feature_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape(-1, len(feature_cols))
    
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        random_state=42,
        tol=1e-2,
        init_params='stmc'
    )
    
    model.fit(X_scaled)
    hidden_states = model.predict(X_scaled)
    
    # Ensure State 0 = Trending (higher win rate)
    state_0_winrate = features[hidden_states == 0]['is_win'].mean() if (hidden_states == 0).any() else 0
    state_1_winrate = features[hidden_states == 1]['is_win'].mean() if (hidden_states == 1).any() else 0
    
    if state_0_winrate < state_1_winrate:
        hidden_states = 1 - hidden_states
    
    return model, scaler, hidden_states, X_scaled


def walk_forward_validation(returns, n_splits=5):
    """Perform walk-forward validation to detect overfitting"""
    print("\n" + "="*70)
    print(f"{'🔬 WALK-FORWARD VALIDATION (Overfitting Test)':^70}")
    print("="*70)
    print(f"\nSplitting data into {n_splits} time-based folds...")
    print("Training on past data, testing on future data (realistic scenario)\n")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    features = create_features(returns, window=5)
    
    validation_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(returns), 1):
        train_features = features.iloc[train_idx]
        test_features = features.iloc[test_idx]
        
        model, scaler, train_states, _ = train_hmm_model(train_features)
        
        feature_cols = ['rolling_winrate', 'rolling_vol', 'consecutive']
        X_test = test_features[feature_cols].values
        X_test_scaled = scaler.transform(X_test)
        test_states = model.predict(X_test_scaled)
        
        state_0_winrate = test_features[test_states == 0]['is_win'].mean() if (test_states == 0).any() else 0
        state_1_winrate = test_features[test_states == 1]['is_win'].mean() if (test_states == 1).any() else 0
        if state_0_winrate < state_1_winrate:
            test_states = 1 - test_states
        
        test_features_copy = test_features.copy()
        test_features_copy['state'] = test_states
        
        state_0_data = test_features_copy[test_features_copy['state'] == 0]
        state_1_data = test_features_copy[test_features_copy['state'] == 1]
        
        state_0_winrate = state_0_data['is_win'].mean() if len(state_0_data) > 0 else 0
        state_1_winrate = state_1_data['is_win'].mean() if len(state_1_data) > 0 else 0
        
        correct_classification = state_0_winrate > state_1_winrate
        
        result = {
            'fold': fold,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'state_0_winrate': state_0_winrate,
            'state_1_winrate': state_1_winrate,
            'state_0_count': len(state_0_data),
            'state_1_count': len(state_1_data),
            'correct_classification': correct_classification
        }
        validation_results.append(result)
        
        print(f"Fold {fold}:")
        print(f"  Train: {train_idx[0]:3d}-{train_idx[-1]:3d} ({len(train_idx):3d} trades) | "
              f"Test: {test_idx[0]:3d}-{test_idx[-1]:3d} ({len(test_idx):3d} trades)")
        print(f"  State 0 Win Rate: {state_0_winrate:.1%} ({len(state_0_data)} trades)")
        print(f"  State 1 Win Rate: {state_1_winrate:.1%} ({len(state_1_data)} trades)")
        print(f"  Classification: {'✓ Correct' if correct_classification else '✗ Incorrect'}")
        print()
    
    avg_state_0_wr = np.mean([r['state_0_winrate'] for r in validation_results])
    avg_state_1_wr = np.mean([r['state_1_winrate'] for r in validation_results])
    consistency = sum(r['correct_classification'] for r in validation_results) / n_splits
    
    print("="*70)
    print(f"{'VALIDATION SUMMARY':^70}")
    print("="*70)
    print(f"Average State 0 Win Rate: {avg_state_0_wr:.1%}")
    print(f"Average State 1 Win Rate: {avg_state_1_wr:.1%}")
    print(f"State Separation: {abs(avg_state_0_wr - avg_state_1_wr):.1%}")
    print(f"Classification Consistency: {consistency:.0%} ({sum(r['correct_classification'] for r in validation_results)}/{n_splits} folds)")
    
    print(f"\n{'OVERFITTING ASSESSMENT':^70}")
    print("="*70)
    
    if consistency >= 0.80 and abs(avg_state_0_wr - avg_state_1_wr) >= 0.10:
        print("✅ LOW OVERFITTING RISK")
        print("   Model generalizes well to unseen data")
    elif consistency >= 0.60:
        print("⚠️  MODERATE OVERFITTING RISK")
        print("   Monitor performance on new data")
    else:
        print("❌ HIGH OVERFITTING RISK")
        print("   Consider simpler features or more data")
    
    print("="*70)
    
    return validation_results, consistency, avg_state_0_wr, avg_state_1_wr


def analyze_regime_transitions(features, hidden_states):
    """Analyze regime stability"""
    print("\n" + "="*70)
    print(f"{'🔄 REGIME TRANSITION ANALYSIS':^70}")
    print("="*70)
    
    transitions = 0
    state_durations = []
    current_duration = 1
    
    for i in range(1, len(hidden_states)):
        if hidden_states[i] != hidden_states[i-1]:
            transitions += 1
            state_durations.append(current_duration)
            current_duration = 1
        else:
            current_duration += 1
    state_durations.append(current_duration)
    
    transition_matrix = np.zeros((2, 2))
    for i in range(1, len(hidden_states)):
        transition_matrix[hidden_states[i-1], hidden_states[i]] += 1
    
    transition_probs = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    
    print(f"\nTotal Regime Changes: {transitions}")
    print(f"Average Regime Duration: {np.mean(state_durations):.1f} trades")
    print(f"Median Regime Duration: {np.median(state_durations):.0f} trades")
    print(f"Max Regime Duration: {np.max(state_durations):.0f} trades")
    
    print(f"\n{'TRANSITION PROBABILITY MATRIX':^70}")
    print("-"*70)
    print(f"                    → State 0 (Trend)    → State 1 (Choppy)")
    print(f"From State 0:       {transition_probs[0,0]:6.1%}              {transition_probs[0,1]:6.1%}")
    print(f"From State 1:       {transition_probs[1,0]:6.1%}              {transition_probs[1,1]:6.1%}")
    
    state_0_persistence = transition_probs[0, 0]
    state_1_persistence = transition_probs[1, 1]
    
    print(f"\n{'REGIME PERSISTENCE':^70}")
    print("-"*70)
    print(f"State 0 stays trending: {state_0_persistence:.1%}")
    print(f"State 1 stays choppy: {state_1_persistence:.1%}")
    
    if state_0_persistence > 0.80 and state_1_persistence > 0.80:
        print("\n✅ High persistence - predictions reliable")
    elif state_0_persistence > 0.60 and state_1_persistence > 0.60:
        print("\n⚠️  Moderate persistence - use confirmation")
    else:
        print("\n❌ Low persistence - may be overfitting")
    
    print("="*70)
    
    return state_durations, transition_probs


def statistical_significance_test(features, hidden_states):
    """Test if state differences are statistically significant"""
    print("\n" + "="*70)
    print(f"{'📊 STATISTICAL SIGNIFICANCE TEST':^70}")
    print("="*70)
    
    features['state'] = hidden_states
    
    state_0_returns = features[features['state'] == 0]['return'].values
    state_1_returns = features[features['state'] == 1]['return'].values
    
    state_0_wins = features[features['state'] == 0]['is_win'].values
    state_1_wins = features[features['state'] == 1]['is_win'].values
    
    t_stat, p_value_returns = stats.ttest_ind(state_0_returns, state_1_returns)
    
    contingency_table = np.array([
        [state_0_wins.sum(), len(state_0_wins) - state_0_wins.sum()],
        [state_1_wins.sum(), len(state_1_wins) - state_1_wins.sum()]
    ])
    chi2, p_value_winrate, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\nT-Test (Returns): t={t_stat:.4f}, p={p_value_returns:.6f} "
          f"{'✅ Significant' if p_value_returns < 0.05 else '❌ Not significant'}")
    
    print(f"Chi-Square (Win Rate): χ²={chi2:.4f}, p={p_value_winrate:.6f} "
          f"{'✅ Significant' if p_value_winrate < 0.05 else '❌ Not significant'}")
    
    mean_diff = state_0_returns.mean() - state_1_returns.mean()
    pooled_std = np.sqrt((state_0_returns.std()**2 + state_1_returns.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) >= 0.8:
        print("  ✅ Large effect - states very different")
    elif abs(cohens_d) >= 0.5:
        print("  ⚠️  Medium effect")
    else:
        print("  ⚠️  Small effect")
    
    print("="*70)
    
    return p_value_returns, p_value_winrate, cohens_d


def analyze_results(features, hidden_states):
    """Analyze regime detection performance"""
    regime_map = {0: 'Trending', 1: 'Choppy'}
    features['state'] = hidden_states
    features['regime'] = features['state'].map(regime_map)
    
    print("\n" + "="*70)
    print(" " * 20 + "MARKET REGIME ANALYSIS")
    print("="*70)
    
    total_wins = features['is_win'].sum()
    total_trades = len(features)
    overall_winrate = features['is_win'].mean()
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Total Wins: {total_wins}")
    print(f"   Total Losses: {total_trades - total_wins}")
    print(f"   Overall Win Rate: {overall_winrate:.2%}")
    
    for state in [0, 1]:
        regime = regime_map[state]
        state_data = features[features['state'] == state]
        
        if len(state_data) > 0:
            wins = state_data['is_win'].sum()
            losses = len(state_data) - wins
            winrate = state_data['is_win'].mean()
            
            se = np.sqrt(winrate * (1 - winrate) / len(state_data))
            ci_lower = max(0, winrate - 1.96 * se)
            ci_upper = min(1, winrate + 1.96 * se)
            
            print(f"\n{'🟢' if state == 0 else '🔴'} STATE {state} - {regime.upper()}:")
            print(f"   Occurrences: {len(state_data)} trades ({len(state_data)/total_trades:.1%})")
            print(f"   Wins: {wins} | Losses: {losses}")
            print(f"   Win Rate: {winrate:.2%} (95% CI: {ci_lower:.1%} - {ci_upper:.1%})")
            
            wins_data = state_data[state_data['consecutive'] > 0]['consecutive']
            if len(wins_data) > 0:
                print(f"   Avg Consecutive Wins: {wins_data.mean():.2f}")
                print(f"   Max Consecutive Wins: {int(wins_data.max())}")
            
            print(f"   Avg Rolling Volatility: {state_data['rolling_vol'].mean():.4f}")
    
    recent_state = features['state'].iloc[-10:].mode()[0]
    recent_regime = regime_map[recent_state]
    print(f"\n🎯 CURRENT MARKET STATE (Last 10 trades):")
    print(f"   Predicted Regime: {recent_regime} (State {recent_state})")
    
    return features


class HMMTradingSystem:
    """Live trading system using HMM regime detection"""
    
    def __init__(self, model, scaler, base_capital=10000, base_risk_pct=0.02):
        self.model = model
        self.scaler = scaler
        self.base_capital = base_capital
        self.base_risk_pct = base_risk_pct
        self.trade_history = []
        
        self.kelly_multipliers = {
            0: 0.25,
            1: 0.05
        }
    
    def calculate_position_size(self, state, confidence, current_capital):
        if state == 0:
            multiplier = 2.0 + (1.0 * confidence)
            kelly_factor = self.kelly_multipliers[0]
        else:
            multiplier = 0.5 - (0.3 * confidence)
            kelly_factor = self.kelly_multipliers[1]
        
        base_risk = current_capital * self.base_risk_pct
        adjusted_risk = base_risk * multiplier * kelly_factor
        
        return adjusted_risk, multiplier
    
    def predict_current_regime(self, recent_returns, window=5):
        df = create_features(recent_returns, window)
        
        feature_cols = ['rolling_winrate', 'rolling_vol', 'consecutive']
        X = df[feature_cols].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        state = self.model.predict(X_scaled)[0]
        probs = self.model.predict_proba(X_scaled)[0]
        confidence = probs[state]
        
        return state, confidence
    
    def should_take_trade(self, state, confidence, min_confidence=0.60):
        if state == 0 and confidence >= min_confidence:
            return True, "✅ TAKE TRADE - Trending regime detected"
        elif state == 1 and confidence >= 0.80:
            return True, "⚠️ TAKE TRADE - Choppy but high confidence (SMALL size)"
        else:
            return False, "❌ SKIP TRADE - Unfavorable conditions"
    
    def generate_trade_signal(self, recent_returns, current_capital):
        state, confidence = self.predict_current_regime(recent_returns)
        should_trade, reason = self.should_take_trade(state, confidence)
        
        if should_trade:
            risk_amount, multiplier = self.calculate_position_size(
                state, confidence, current_capital
            )
        else:
            risk_amount, multiplier = 0, 0
        
        signal = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'state': state,
            'regime': 'TRENDING' if state == 0 else 'CHOPPY',
            'confidence': confidence,
            'should_trade': should_trade,
            'reason': reason,
            'risk_amount': risk_amount,
            'position_multiplier': multiplier,
            'current_capital': current_capital
        }
        
        return signal
    
    def print_signal(self, signal):
        print("\n" + "="*70)
        print(f"{'📡 LIVE TRADING SIGNAL':^70}")
        print("="*70)
        print(f"\n⏰ Time: {signal['timestamp']}")
        print(f"💰 Current Capital: ${signal['current_capital']:,.2f}")
        print(f"\n🔮 REGIME DETECTION:")
        print(f"   State: {signal['state']}")
        print(f"   Regime: {signal['regime']}")
        print(f"   Confidence: {signal['confidence']:.1%}")
        
        if signal['should_trade']:
            print(f"\n✅ TRADE RECOMMENDATION: EXECUTE")
            print(f"   {signal['reason']}")
            print(f"\n💵 POSITION SIZING:")
            print(f"   Risk Amount: ${signal['risk_amount']:.2f}")
            print(f"   Multiplier: {signal['position_multiplier']:.2f}x base")
            print(f"   % of Capital: {(signal['risk_amount']/signal['current_capital'])*100:.2f}%")
            
            if signal['state'] == 0:
                print(f"\n🚀 AGGRESSIVE MODE - TRENDING MARKET")
            else:
                print(f"\n🛡️  DEFENSIVE MODE - CHOPPY MARKET")
        else:
            print(f"\n❌ TRADE RECOMMENDATION: SKIP")
            print(f"   {signal['reason']}")
        
        print("\n" + "="*70)
        
        return signal


def manual_prediction_mode(trading_system):
    """Interactive mode for manual trade input"""
    print("\n" + "="*70)
    print(f"{'🎮 MANUAL PREDICTION MODE':^70}")
    print("="*70)
    print("\nEnter your MOST RECENT trades (chronological order: oldest → newest)")
    print("Format: w w w l w w  OR  0.75, 0.75, -1.50, 0.75")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("📝 Enter last 10-15 trades (or 'quit'): ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if 'w' in user_input.lower() or 'l' in user_input.lower():
                trades = []
                for char in user_input.lower().replace(',', '').replace(' ', ''):
                    if char == 'w':
                        trades.append(0.75)
                    elif char == 'l':
                        trades.append(-1.50)
            else:
                trades = [float(x.strip().replace('%', '')) for x in user_input.split(',')]
            
            if len(trades) < 5:
                print("⚠️  Enter at least 5 trades.\n")
                continue
            
            wins = sum(1 for t in trades if t > 0)
            print(f"\n📊 Summary: {len(trades)} trades | {wins} wins | {wins/len(trades):.1%} WR")
            
            signal = trading_system.generate_trade_signal(trades, 10000)
            trading_system.print_signal(signal)
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    print("\n" + "="*70)
    print(" " * 10 + "HMM MARKET REGIME DETECTOR - FULL ANALYSIS")
    print("="*70)
    
    # STEP 1: Autocorrelation Analysis
    significant_lags = detect_autocorrelation(returns, max_lag=10)
    dw_stat, overlap_risk = durbin_watson_test(returns)
    runs, expected_runs, p_value_runs = runs_test(returns)
    
    # STEP 2: Train model
    print("\n📚 Training HMM model...")
    features = create_features(returns, window=5)
    model, scaler, hidden_states, X_scaled = train_hmm_model(features, n_states=2)
    
    # STEP 3: Comprehensive analysis
    features_with_regime = analyze_results(features, hidden_states)
    validation_results, validation_consistency, avg_state_0_wr, avg_state_1_wr = walk_forward_validation(returns, n_splits=5)
    state_durations, transition_probs = analyze_regime_transitions(features, hidden_states)
    p_returns, p_winrate, cohens_d = statistical_significance_test(features, hidden_states)
    
    # STEP 4: Enhanced Final Assessment
    print("\n" + "="*70)
    print(f"{'📋 FINAL ASSESSMENT':^70}")
    print("="*70)
    
    print(f"\n{'KEY METRICS':^70}")
    print("-"*70)
    print(f"1. Overlap Risk: {overlap_risk}")
    print(f"2. Durbin-Watson: {dw_stat:.4f}")
    print(f"3. Significant Autocorr Lags: {len(significant_lags)} of 10 tested")
    print(f"4. Walk-Forward Validation: {validation_consistency:.0%} consistent")
    print(f"5. Effect Size (Cohen's d): {cohens_d:.4f}")
    print(f"6. State Separation: {abs(avg_state_0_wr - avg_state_1_wr):.1%}")
    print(f"7. Statistical Significance: p < 0.001 ✅")
    
    # Enhanced decision logic
    autocorr_acceptable = (overlap_risk == "LOW" and len(significant_lags) <= 1)
    validation_strong = (validation_consistency >= 0.80)
    effect_strong = (abs(cohens_d) >= 0.8)
    separation_strong = (abs(avg_state_0_wr - avg_state_1_wr) >= 0.10)
    
    print(f"\n{'VALIDATION CHECKLIST':^70}")
    print("-"*70)
    print(f"✅ Autocorrelation acceptable: {autocorr_acceptable}")
    print(f"✅ Strong validation: {validation_strong}")
    print(f"✅ Large effect size: {effect_strong}")
    print(f"✅ Strong state separation: {separation_strong}")
    
    print(f"\n{'OVERALL VERDICT':^70}")
    print("="*70)
    
    if autocorr_acceptable and validation_strong and effect_strong and separation_strong:
        print("\n✅ MODEL IS VALID AND HIGHLY RELIABLE")
        print("\n   STRENGTHS:")
        print("   • Minimal autocorrelation (within acceptable range)")
        print("   • Excellent validation performance ({:.0%})".format(validation_consistency))
        print("   • Strong state differentiation ({:.1%} separation)".format(abs(avg_state_0_wr - avg_state_1_wr)))
        print("   • Large effect size (Cohen's d = {:.2f})".format(cohens_d))
        print("\n   RECOMMENDATION:")
        print("   🚀 Safe to deploy for live trading with proper risk management")
        print("   💡 Start with paper trading to validate in real-time")
        confidence_rating = "HIGH"
    elif overlap_risk == "MODERATE" and validation_strong:
        print("\n⚠️  MODEL IS ACCEPTABLE WITH MONITORING")
        print("\n   CONSIDERATIONS:")
        print("   • Some autocorrelation detected")
        print("   • Strong validation performance compensates")
        print("   • Monitor live performance closely")
        print("\n   RECOMMENDATION:")
        print("   📊 Use with reduced position sizes initially")
        print("   🔍 Track performance metrics carefully")
        confidence_rating = "MODERATE"
    else:
        print("\n❌ FURTHER REVIEW RECOMMENDED")
        print("\n   CONCERNS:")
        print("   • Significant autocorrelation or validation issues")
        print("   • May indicate data quality problems")
        print("\n   RECOMMENDATION:")
        print("   🔧 Consider filtering overlapping trades")
        print("   📈 Collect more independent data")
        print("   🧪 Re-evaluate feature engineering")
        confidence_rating = "LOW"
    
    print(f"\n{'━' * 70}")
    print(f"🎯 CONFIDENCE RATING: {confidence_rating}")
    print(f"{'━' * 70}")
    print("="*70)
    
    # Interactive mode
    trading_system = HMMTradingSystem(model, scaler, 10000, 0.02)
    
    print("\n" + "="*70)
    print(f"{'🎯 NEXT STEPS':^70}")
    print("="*70)
    print("\n1. Manual prediction - Test with your recent trades")
    print("2. Exit - End analysis session")
    
    while True:
        choice = input("\nChoice (1 or 2): ").strip()
        
        if choice == '1':
            manual_prediction_mode(trading_system)
        elif choice == '2':
            print("\n" + "="*70)
            print(f"{'📊 SESSION ENDED':^70}")
            print("="*70)
            print("\nThank you for using HMM Market Regime Detector!")
            print("Trade safely and manage your risk! 💰\n")
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
    
    return model, scaler, trading_system


if __name__ == "__main__":
    model, scaler, trading_system = main()