# Домашнее задание к уроку 3: Полносвязные сети
## Задание 1: Эксперименты с глубиной сети (30 баллов)

### 1.1 Сравнение моделей разной глубины (15 баллов)
Создал и обучил модели с различным количеством слоев:
- 1 слой (линейный классификатор) | (train acc - 0.9842; test acc - 0.9756; training time - 74.38 sec)
- 2 слоя (1 скрытый) | (train acc - 0.9841; test acc - 0.9749; training time - 77.07)
- 3 слоя (2 скрытых) | (train acc - 0.9828; test acc - 0.9716; training time - 79.47)
- 5 слоев (4 скрытых) | (train acc - 0.9801; test acc - 0.9737; training time - 79.22)
- 7 слоев (6 скрытых) | (train acc - 0.979; test acc - 0.9769; training time - 80.79)

Модель с 1 слоем:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/depth_1.png)

Модель с 2 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/depth_2.png)

Модель с 3 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/depth_3.png)

Модель с 5 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/depth_5.png)

Модель с 7 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/depth_7.png)

Из графиков видно, что обучение было нестабильным, скорее всего из-за высокого learning rate. Лучший результат показала модель с 7 слоями (acc = 0.9769)

### 1.2 Анализ переобучения (15 баллов)
Исследовал влияние глубины на переобучение:
- Построил графики train/test accuracy по эпохам
- Добавил Dropout и BatchNorm, результаты оказались похожие.

Модель с 1 слоем:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/overfitting_depth_1.png)

Модель с 2 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/overfitting_depth_2.png)

Модель с 3 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/overfitting_depth_3.png)

Модель с 5 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/overfitting_depth_5.png)

Модель с 7 слоями:
![Layers](https://github.com/4pokodav/lesson_3/raw/main/plots/overfitting_depth_7.png)

Из графиков видно, что переобучения не получилось достичь, следовательно, модели оказались устойчивы к переобучению.

## Задание 2: Эксперименты с шириной сети (25 баллов)

### 2.1 Сравнение моделей разной ширины (15 баллов)
Создал модели с различной шириной слоев, но с фиксированной глубиной (3 слоя):
- Узкие слои: [64, 32, 16] | (train acc - 0.9759; test acc - 0.9614; training time - 77.58 sec)
- Средние слои: [256, 128, 64] | (train acc - 0.9867; test acc - 0.9786; training time - 76.91 sec)
- Широкие слои: [1024, 512, 256] | (train acc - 0.9866; test acc - 0.9783; training time - 99.98 sec)
- Очень широкие слои: [2048, 1024, 512] | (train acc - 0.982; test acc - 0.9765; training time - 124.36 sec)

Модель с узкими слоями:
![Layers width](https://github.com/4pokodav/lesson_3/raw/main/plots/width_narrow.png)

Модель с средними слоями:
![Layers width](https://github.com/4pokodav/lesson_3/raw/main/plots/width_medium.png)

Модель с широкими слоями:
![Layers width](https://github.com/4pokodav/lesson_3/raw/main/plots/width_wide.png)

Модель с очень широкими слоями:
![Layers width](https://github.com/4pokodav/lesson_3/raw/main/plots/width_very_wide.png)

Количество параметров возрастает примерно в 4-5 раз у следующих моделей. 
У первой модели 50 тыс. параметров, а у последней модели 4.5 млн параметров.
При этом заметно, что ширина слоев не сильно увеличила точность.
Наилучший результат у модели с средними слоями: [256, 128, 64] (acc = 0.9786)

### 2.2 Оптимизация архитектуры (10 баллов)

Найдите оптимальную архитектуру:
- Использовал grid search для поиска лучшей комбинации
- Визуализиализировал результаты в виде heatmap

![Heatmap](https://github.com/4pokodav/lesson_3/raw/main/plots/grid_search_heatmap.png)

Анализ heatmap:
- Увеличение ширины обоих слоёв улучшает метрику (Самые высокие значения наблюдаются при больших Layer 1 и Layer 2)
- Самые высокие значения наблюдаются при больших Layer 1 и Layer 2 (256 × 128).
- Маленькие ширины слоёв приводят к плохим результатам:
- Рост Layer 1 даёт более заметное улучшение, чем Layer 2.

## Задание 3: Эксперименты с регуляризацией (25 баллов)

### 3.1 Сравнение техник регуляризации (15 баллов)

Исследовал различные техники регуляризации:
- Без регуляризации
- Только Dropout (разные коэффициенты: 0.1, 0.3, 0.5)
- Только BatchNorm
- Dropout + BatchNorm
- L2 регуляризация (weight decay)

Без регуляризации:
![Accuracy](https://github.com/4pokodav/lesson_3/raw/main/plots/No_Regularization_history.png)
![Weights](https://github.com/4pokodav/lesson_3/raw/main/plots/No_Regularization_weights_hist.png)

Только Dropout:
Dropout = 0.1
![Accuracy](https://github.com/4pokodav/lesson_3/raw/main/plots/Dropout_0.1_history.png)
![Weights](https://github.com/4pokodav/lesson_3/raw/main/plots/Dropout_0.1_weights_hist.png)

Dropout = 0.3
![Accuracy](https://github.com/4pokodav/lesson_3/raw/main/plots/Dropout_0.3_history.png)
![Weights](https://github.com/4pokodav/lesson_3/raw/main/plots/Dropout_0.3_weights_hist.png)

Dropout = 0.5
![Accuracy](https://github.com/4pokodav/lesson_3/raw/main/plots/Dropout_0.5_history.png)
![Weights](https://github.com/4pokodav/lesson_3/raw/main/plots/Dropout_0.5_weights_hist.png)

Только BatchNorm:
![Accuracy](https://github.com/4pokodav/lesson_3/raw/main/plots/BatchNorm_only_history.png)
![Weights](https://github.com/4pokodav/lesson_3/raw/main/plots/BatchNorm_only_weights_hist.png)

Dropout + BatchNorm:
![Accuracy](https://github.com/4pokodav/lesson_3/raw/main/plots/Dropout_0.5_+_BatchNorm_history.png)
![Weights](https://github.com/4pokodav/lesson_3/raw/main/plots/Dropout_0.5_+_BatchNorm_weights_hist.png)

L2 регуляризация:
![Accuracy](https://github.com/4pokodav/lesson_3/raw/main/plots/L2_regularization_(1e-4)_history.png)
![Weights](https://github.com/4pokodav/lesson_3/raw/main/plots/L2_regularization_(1e-4)_weights_hist.png)

Наилучший результат показала модель x (acc = x)

### 3.2 Адаптивная регуляризация (10 баллов)

Реализовал адаптивные техники:
- Dropout с изменяющимся коэффициентом
- BatchNorm с различными momentum
- Комбинирование нескольких техник

Dropout с изменяющимся коэффициентом:
![Accuracy](https://github.com/4pokodav/lesson_3/raw/main/plots/Adaptive_Dropout_+_BatchNorm_momentum=0.1_history.png)
![Weights](https://github.com/4pokodav/lesson_3/raw/main/plots/Adaptive_Dropout_+_BatchNorm_momentum=0.1_weights_hist.png)

BatchNorm с различными momentum:
![Accuracy](https://github.com/4pokodav/lesson_3/raw/main/plots/Adaptive_Dropout_+_BatchNorm_momentum=0.5_history.png)
![Weights](https://github.com/4pokodav/lesson_3/raw/main/plots/Adaptive_Dropout_+_BatchNorm_momentum=0.5_weights_hist.png)

Комбинирование нескольких техник:
![Accuracy](https://github.com/4pokodav/lesson_3/raw/main/plots/Adaptive_Dropout_+_BatchNorm_momentum=0.9_history.png)
![Weights](https://github.com/4pokodav/lesson_3/raw/main/plots/Adaptive_Dropout_+_BatchNorm_momentum=0.9_weights_hist.png)

Наилучший результат показала модель x (acc = x)
