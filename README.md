# ВШЭ ПСБ.Хак 🚀

По результатам очных защит заняли 11 место в общем зачете из 100 команд, принявших участие в [хакатоне](https://ai.hse.ru/hacks/psb24). 

- **Task**: `Binary classification`
- **Metric**: `ROC-AUC`
- **Best score**: `0.8511`

### 💻 Техническая часть
Были опробованы разные подходы: бустинги ([LGBM](https://lightgbm.readthedocs.io/en/stable/) & [Catboost](https://github.com/catboost/catboost/tree/master)), полносвязная нейронная сеть, блендинг (добавление предсказаний FCNn как признак в бустинговые модельки), а также ансамблирование полученных решений. Весь код моделей представлен в jupyter ноутбуках `PSBxHSE_hack_Model_NN.ipynb` и `PSBxHSE_hack_Model.ipynb`. Для подбора гиперпараметров модели использовалась `Optuna`. В FCNn добавили инициализацию весов *Xavier*, dropouts и батч-нормализацию

Из имевшихся признаков были сгенерированы новые на основе дат:
- номер дня, недели, месяца в году
- кол-во дней с даты брони до заезда
- флаги на праздники, выходные и будние дни
- *sin* и *cos* от дат для паттернов сезонности и др.

Обработаны категориальные признаки, рассчитаны агрегаты из информации о стоимости в бронированиях, а также по API от Visualcrossing добавлены внешние фичи (температура на каждую дату, скорость и направление ветра, осадки, давление, фаза луны 🌙 и прочее)

![ScreenShot](https://github.com/ez3nx/HSExPSB_hackaton/blob/main/HSE_PSB_Hack_GlowByte_ml.jpg)

![ScreenShot](https://github.com/ez3nx/HSExPSB_hackaton/blob/main/HSE_PSB_Hack_GlowByte_MLOps.jpg)
