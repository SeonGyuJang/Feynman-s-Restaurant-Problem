import numpy as np
from scipy.special import expit
from scipy import integrate
from typing import List, Tuple, Dict

# Define the threshold policy class
threshold_policy_class = "linear_by_proportion_of_nights_remaining"

def uniform_rating(u): return u
def exponential_rating(u): return -np.log(1 - u) * 0.5
def power_law_rating(u): return np.sqrt(1 / (1 - u)) * 0.25
def triangular_rating(u): return np.sqrt(u) * 0.75

rating_functions = {
    'uniform': uniform_rating,
    'exponential': exponential_rating,
    'power_law': power_law_rating,
    'triangular': triangular_rating
}

# Probability Density Functions
def uniform_pdf(x): return 1 if 0 <= x <= 1 else 0
def exponential_pdf(x): return 2 * np.exp(-2 * x) if x >= 0 else 0
def power_law_pdf(x): return 1 / (8 * x**3) if x > 0.25 else 0
def triangular_pdf(x): return 32 * x / 9 if 0 <= x <= 0.75 else 0

# Probability Mass Functions
def uniform_pmf():
    scores = list(range(101))
    probs = [integrate.quad(uniform_pdf, k/101, (k+1)/101)[0] for k in scores]
    return scores, probs

def exponential_pmf(max_rating=2_000):
    scores = list(range(max_rating + 1))
    probs = [integrate.quad(exponential_pdf, k/101, (k+1)/101)[0] for k in scores]
    return scores, probs

def power_law_pmf(max_rating=100_000):
    scores = list(range(25, max_rating + 1))
    probs = [integrate.quad(power_law_pdf, k/101, (k+1)/101)[0] for k in scores]
    return scores, probs

def triangular_pmf():
    scores = list(range(76))
    probs = [integrate.quad(triangular_pdf, k/101, (k+1)/101)[0] for k in scores]
    return scores, probs

# Create PMFs for each distribution
uniform_pmf_values = uniform_pmf()
exponential_pmf_values = exponential_pmf()
power_law_pmf_values = power_law_pmf()
triangular_pmf_values = triangular_pmf()
rating_pmfs = {
    'uniform': uniform_pmf_values,
    'exponential': exponential_pmf_values,
    'power_law': power_law_pmf_values,
    'triangular': triangular_pmf_values
}

# Define the thresholds for each night
def generate_thresholds(threshold_policy_class, nights, slope, intercept):
    if threshold_policy_class == "linear_by_proportion":
        return [slope * (night / nights) + intercept for night in range(nights)]
    elif threshold_policy_class == "linear_by_proportion_of_nights_remaining":
        return [slope * ((nights - night) / nights) + intercept for night in range(nights)]
    elif threshold_policy_class == "linear_by_nights_remaining":
        return [slope * (nights - night) + intercept for night in range(nights)]
    elif threshold_policy_class == "linear_by_nights_elapsed":
        return [slope * night + intercept for night in range(nights)]
    else:
        raise ValueError("Threshold policy class not supported")

def expected_value(nights_remaining, distribution, best_restaurant_rating, thresholds, cache, power_law_max=2500, exponential_max=500):
    # print thresholds
    if nights_remaining == 0:
        return 0
    if (nights_remaining, best_restaurant_rating) in cache:
        return cache[(nights_remaining, best_restaurant_rating)]
    # TODO: update cache to include thresholds in the future (past is irrelevant) and possibly distro too,
    # so that we can use it across trials

   # Decision logic
    if best_restaurant_rating < thresholds[-nights_remaining]:
        scores, probs = rating_pmfs[distribution]
        if distribution == 'power_law':
            # Slice them to the max rating
            scores, probs = scores[:power_law_max], probs[:power_law_max]
        elif distribution == 'exponential':
            scores, probs = scores[:exponential_max], probs[:exponential_max]
        expected_values = np.array([expected_value(nights_remaining - 1, distribution, max(best_restaurant_rating, score), thresholds, cache, power_law_max, exponential_max) for score in scores])
        value = np.dot(probs, scores + expected_values)
    else:
        value = best_restaurant_rating + expected_value(nights_remaining - 1, distribution, best_restaurant_rating, thresholds, cache, power_law_max, exponential_max)

    cache[(nights_remaining, best_restaurant_rating)] = value
    return value

def expected_value_optimized(nights_remaining: int, distribution: str, best_restaurant_rating: float, thresholds: List[float], 
                             cache: Dict[Tuple[int, float, str], float], power_law_max: int = 2500, exponential_max: int = 500) -> float:
    if nights_remaining == 0:
        return 0

    cache_key = (nights_remaining, best_restaurant_rating, distribution)
    if cache_key in cache:
        return cache[cache_key]

    if best_restaurant_rating < thresholds[-nights_remaining]:
        scores, probs = rating_pmfs[distribution]
        if distribution == 'power_law':
            scores, probs = scores[:power_law_max], probs[:power_law_max]
        elif distribution == 'exponential':
            scores, probs = scores[:exponential_max], probs[:exponential_max]

        future_values = np.array([
            expected_value_optimized(nights_remaining - 1, distribution, max(best_restaurant_rating, score), thresholds, cache, power_law_max, exponential_max)
            for score in scores
        ])
        value = np.dot(probs, scores + future_values)
    else:
        value = best_restaurant_rating + expected_value_optimized(nights_remaining - 1, distribution, best_restaurant_rating, thresholds, cache, power_law_max, exponential_max)

    cache[cache_key] = value
    return value

def expected_value_with_noise(nights_remaining, beta, distribution, best_restaurant_rating, thresholds, cache, power_law_max=100_000, exponential_max=2_000):
    if nights_remaining == 0:
        return 0
    if (nights_remaining, best_restaurant_rating) in cache:
        return cache[(nights_remaining, best_restaurant_rating)]

    night = len(thresholds) - nights_remaining

    scores, probs = rating_pmfs[distribution]
    if distribution == 'power_law':
        # Slice them to the max rating
        scores, probs = scores[:power_law_max], probs[:power_law_max]
    elif distribution == 'exponential':
        scores, probs = scores[:exponential_max], probs[:exponential_max]

    explore_value = sum((score + expected_value_with_noise(nights_remaining - 1, beta, distribution, max(best_restaurant_rating, score), thresholds, cache, power_law_max, exponential_max)) * probs[i] for i, score in enumerate(scores))
    if night == 0:
        value = explore_value
    else:
        probability_of_exploring = expit(beta * (thresholds[night] - best_restaurant_rating))
        exploit_value = best_restaurant_rating + expected_value_with_noise(nights_remaining - 1, beta, distribution, best_restaurant_rating, thresholds, cache, power_law_max, exponential_max)
        value = probability_of_exploring * explore_value + (1-probability_of_exploring) * exploit_value

    cache[(nights_remaining, best_restaurant_rating)] = value
    return value