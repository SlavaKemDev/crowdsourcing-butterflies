# Запуск

Все команды выполняются из корня репозитория.

## Установка

Нужен Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Данные и метрики

[Манифест](data/manifest.jsonl) хранит ссылки и контрольные суммы 400 фото. [Эталон](data/golden.jsonl) содержит рамки. Чтобы восстановить фотографии, понадобятся `curl` и доступ к источникам:

```bash
.venv/bin/python dataset.py collect
```

Для пересчёта метрик фотографии и запуск моделей не нужны:

```bash
for run in dino dino-threshold035 owlv2 qwen; do
  .venv/bin/python metrics.py evaluate \
    --golden data/golden.jsonl \
    --manifest data/manifest.jsonl \
    --predictions "runs/$run/predictions.jsonl" \
    --output "/tmp/hw5-$run-metrics.json"
done
```

## Повтор экспериментов

У Grounding DINO сравнили пороги 0,25 и 0,35. Гипотеза: повышение порога уберёт часть ложных рамок, но может добавить пропуски. Вариант с порогом 0,35 можно получить из сохранённых кандидатов:

```bash
.venv/bin/python models.py postprocess \
  --predictions runs/dino/predictions.jsonl \
  --threshold 0.35 --output /tmp/hw5-dino-035.jsonl
```

Версии весов, промпты и настройки сохранены в `runs/*/predictions.run.json`. Для повтора Qwen на Mac с MPS сначала восстановите фотографии:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python models.py infer \
  --model qwen --device mps --manifest data/manifest.jsonl \
  --output /tmp/hw5-qwen-reproduction.jsonl
```

## Разметка

[Проект разметки](https://tasks.yandex.ru/requester/project/11858) · [Проект проверки](https://tasks.yandex.ru/requester/project/11859)

Настройки проектов и допуска лежат в `prepared/`. Автор не мог проверять собственную разметку. Две независимые приёмки позволяли принять ответ, два отклонения вели к исправлению, расхождения разбирали по кадру. Шесть предварительных решений отмечены в [сводке](report/annotation_summary.json).
