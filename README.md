---
title: "forec_methods_notebook"
author: "Inês Chainho"
date: "2025-05-08"
output: pdf_document
---

### Introduction

Recently, energy forecasting has become increasingly important for supporting grid stability and sustainable energy policy. Accurate forecasts are essential for energy providers to anticipate demand, manage resources efficiently, and support the ongoing energy transition, so we will focus on forecasting electricity consumption in Portugal. The goal of this project is to apply the Box-Jenkins methodology to model and forecast electricity usage patterns, helping to inform decisions related to grid management and energy planning.

The data we will use, from 1980 to 2021, was withdrawn from a Global Electricity Statistics dataset in Kaggle, where only the data relating to Portugal was kept. The dataset is in the normal Time Series format, where the index is the years and the features are the following variables:

* Elec_generation - Electricity generation/production in billion kWh;
* Elec_consumption - Electricity consumption in billion kWh;
* Elec_imports - Electricity imports in billion kWh;
* Elec_exports - Electricity exports in billion kWh;
* Net_imports - Electricity net imports in billion kWh;
* Installed_capacity - The maximum amount of electricity that a generating station can produce in million kW;
* Distribution_loss - Losses that occur in transmission of electricity between the sources of supply and points of distribution in billion kWh.

### Stationarity of the process

Firstly, let's open all the libraries needed:
```
library(fpp3)
library(tseries)
library(urca)
library(forecast)
library(fable)
```

Now, before anything else, we need to make sure that the variable we are going to study - electricity consumption - belongs to a stochastic stationary process.

In the next piece of code, we are going to import the dataset and create a tsibble off of it, with the index being the time variable - X. Then, we will autoplot the desired variable - Elec_consumption.

```
# original dataset
eletricity_ds <- read.csv(file.choose())
# read.csv("C:/Users/BeatrizGil/Documents/Métodos de #Previsão/forecasting-projet/Portugal_yearly_energy_data.csv")
View(eletricity_ds)

# original tsibble
elec_tsibble <- eletricity_ds %>% as_tsibble(index = X)
View(elec_tsibble)
elec_tsibble %>% autoplot(Elec_consumption)
```

In the next part, we will make an Augmented Dickey Fuller Test to check the stationarity of the process chosen. We proceeded by creating an array of the values in the desired variable - Elec_consumption - and setting "start" to the first year considered - 1980 - and "frequency" to 1, since it's yearly data. If the p-value < 0.05, we will reject the null hypothesis and conclude that the series is stationary. If the p-value ≥ 0.05, the series is non-stationary.
We will also display the plot, the ACF, and the PACF so we can get some insights.

```
set.seed(30)
# Elec_consumption original array
data <- eletricity_ds %>% select(Elec_consumption)

# Elec_consumption time series array
data_ts <- ts(data, start = 1980, frequency = 1)
adf_result <- adf.test(data_ts)
print(adf_result)

# plotting the Elec_consumption, ACF and PACF
elec_tsibble %>% gg_tsdisplay(Elec_consumption, plot_type='partial')
```

Since the p-value = 0.9739, we accepted the null hypotheses, meaning the series is non-stationary. This is also clearly visible in the ACF function, which is infinite but doesn't decay to zero exponentially.
By looking at the Elec_consumption plot, it looks like a random walk, but let's check if it actually is by doing the KPSS Unit Root Test.

```
kpss_test <- ur.kpss(data_ts)
summary(kpss_test)
```

The test statistic is greater than all critical values, so we reject the null hypotheses and the time series is very likely difference-stationary (DSP).

```
# First-Order Differenced Elec_consumption time series array
ts_diff <- diff(data_ts, differences = 1)
plot(ts_diff, main = "Differenced Electricity Consumption")
kpss_test2 <- ur.kpss(ts_diff)
summary(kpss_test2)
adf_result2 <- adf.test(ts_diff)
print(adf_result2)
```

By making the same tests as before, we conclude the first order differed time series is still non-stationary, so we will difference again.

```
# Second-Order Differenced Elec_consumption time series array
ts_diff2 <- diff(ts_diff, differences = 1)

plot(ts_diff2, main = "Second Order Differenced Electricity Consumption")
kpss_test3 <- ur.kpss(ts_diff2)
summary(kpss_test3)
adf_result3 <- adf.test(ts_diff2)
print(adf_result3)
```

Now, we finally have a stationary time series relating to the Electricity Consumption in the period considered.

### Tentative Identification

In the next piece of code we will generate the ACF and the PACF of the second order differenced time series, so we can suggest some candidate models for analysis during the next stages.

```
# Complete Second-Order Differenced Elec_consumption time series array
ts_diff_comp <- c(NA, NA, ts_diff2)

# Complete original tsibble
elec_tsibble_comp <- elec_tsibble %>% mutate(Elec_cons_diff = ts_diff_comp)
View(elec_tsibble_comp)
elec_tsibble_comp %>% gg_tsdisplay(Elec_cons_diff, plot_type='partial')
```

By looking at the ACF, we can see it is finite and cuts off after lag 1 or 2, which suggests MA(1) or MA(2) model. By looking at the PACF, we can see it is infinite and exponentially decays to 0 after lag 2. The models we suggest looking into are ARIMA(2, 2, 1) and ARIMA(2, 2, 2).
Let's also see what the algorithm picks:

```
auto.arima(ts_diff2)
```
The algorithm suggests ARIMA(0,0,1) so we will also look into ARIMA(0, 2, 1).
Now let's create a function to calculate the Information Criteria through the 3 equations we know - Akaike IC (AIC), Hannan-Quinn IC (HQIC) and Bayesian or Schwarz IC (BSIC) - and test the model who minimizes them the most.

```
calc_ic <- function(dev, p, q, T) {
  ldev <- log(dev)
  k <- p + q + 1
  aic <- ldev + 2*k/T
  hqic <- ldev + 2*k*log(log(T))/T
  bsic <- ldev + k*log(T)/T
  return(c(aic, hqic, bsic))}

# fit ARIMA(2,2,1)
model221 <- Arima(data_ts, order = c(2,2,1))
dev221 <- model221$sigma2
T221 <- length(model221$residuals)
calc_ic(dev221, 2, 1, T221)

# fit ARIMA(2,2,2)
model222 <- Arima(data_ts, order = c(2,2,2))
dev222 <- model222$sigma2
T222 <- length(model222$residuals)
calc_ic(dev222, 2, 2, T222)

# fit ARIMA(0,2,1)
model021 <- Arima(data_ts, order = c(0,2,1))
summary(model021)
dev021 <- model021$sigma2
T021 <- length(model021$residuals)
calc_ic(dev021, 0, 1, T021)
```
Acording to the values obtained, the model that minimizes the Information Criteria is ARIMA(0,2,1), so this should be the model we use from now on.

### Estimation

Firstly, we will fit the model and check its report and residuals.

```
elec_tsibble %>% model(ARIMA(Elec_consumption ~ pdq(0,2,1))) %>% report()
elec_tsibble %>% select(X, Elec_consumption) %>% select(arima021) %>% gg_tsresiduals()
```

Now we will do the Ljung-Box and the Ljung-Box-Pierce methods to check if the model is adequate or not with an hypotheses test:

```
# Fitted original tsibble
elec_fit <- elec_tsibble %>% model(arima_021 = ARIMA(Elec_consumption ~ pdq(0,2,1)))

View(elec_fit)
augment(elec_fit) %>% features(.innov, ljung_box, lag = 10, dof = 3)
augment(elec_fit) %>% features(.innov, box_pierce, lag = 10, dof = 3)
```

Since the p-value (0.898 and 0.846) is greater than the significance level (0.05), we accept the null hypotheses, meaning there is no residual autocorrelation, and our model is adequate.

### Forecasting Implementation

Now that we have identified and validated the ARIMA(0,2,1) model as the best candidate for modeling electricity consumption in Portugal, we can proceed with the forecasting phase. This step involves generating future values based on the fitted model and analyzing the predicted trends. The idea is to understand how electricity consumption may evolve in the upcoming years, assuming past patterns continue.

```
# Forecasting 10 years ahead using the fitted ARIMA(0,2,1) model
forecast_elec <- elec_fit %>% forecast(h = "10 years")

# Plot forecast with historical data
forecast_elec %>% autoplot(elec_tsibble) +
  labs(title = "Electricity Consumption Forecast (Portugal)",
       y = "Electricity Consumption (billion kWh)",
       x = "Year")
```

### Accuracy Assessment

To validate the predictive ability of our chosen model, we split the dataset into training and testing subsets. The training data was used to fit the model, and the last five years were kept as a test set to evaluate forecast performance. This approach allows us to assess how well the model generalizes to unseen data using established accuracy metrics.

```
# Split into training/testing sets for accuracy 
train_data <- elec_tsibble %>% filter(X <= 2016)
test_data  <- elec_tsibble %>% filter(X > 2016)

# Fit model on training set
train_fit <- train_data %>% model(arima_train = ARIMA(Elec_consumption ~ pdq(0,2,1)))

# Forecast 5 years ahead and compare with actual values
fc <- train_fit %>% forecast(h = "5 years")

# Plot forecast vs actual
fc %>% autoplot(train_data, level = NULL) +
  autolayer(test_data, Elec_consumption, color = "red") +
  labs(title = "Forecast vs Actual: ARIMA(0,2,1)",
       y = "Electricity Consumption",
       x = "Year")

# Forecast accuracy
fc %>% accuracy(test_data)
```
The results show strong predictive performance:
- The model had a low mean error (ME = 0.190), suggesting minimal bias.
- Both RMSE (0.811) and MAE (0.731) indicate a good fit, with relatively small forecast deviations.
- A MAPE of 1.50% confirms excellent overall accuracy, with predictions deviating only slightly from actual consumption values.
- Residual autocorrelation was low (ACF1 = 0.291), indicating that most of the temporal structure was successfully captured.
- Although some metrics (MASE and RMSSE) returned NaN values, likely due to test sample size or scaling limitations, the available indicators suggest that the model performs well.

### Conclusion and Reflection

In this project, we applied the Box-Jenkins methodology to model and forecast electricity consumption in Portugal using annual data from 1980 to 2021. After testing for stationarity using the ADF and KPSS tests, we concluded that the series was second-order difference stationary. Through ACF/PACF diagnostics and information criteria comparisons, ARIMA(0,2,1) was selected as the most appropriate model.

The model performed adequately, as confirmed by the Ljung-Box test showing no significant autocorrelation in the residuals. Forecasts produced by the model indicate a continued upward trend in electricity consumption, although with some uncertainty due to the differencing applied and limited frequency of the data (annual). Accuracy was assessed using a 5-year holdout, and metrics like RMSE and MAPE showed good performance, though the small size of the test set limits the robustness of these results.

Nonetheless, there are some limitations to consider. The use of annual data reduces the granularity of the analysis, meaning potential seasonal patterns cannot be captured. Additionally, external shocks, such as policy changes, the COVID-19 pandemic or energy market disruptions are not explicitly accounted for in the model.

Possible improvements could involve using a higher-frequency dataset (e.g., monthly or quarterly), which would allow the implementation of models that handle seasonality, such as SARIMA or ETS. Moreover, machine learning methods like neural network-based models could be investigated to enhance forecast accuracy and capture more complex patterns when richer datasets are accessible.
