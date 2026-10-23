# AGENTS.md — Инструкция и контекст для AI-агентов

Этот документ служит входной точкой для AI-ассистентов и агентов, работающих в данном репозитории.
Его главная цель — **не делать одну и ту же работу дважды, опираться на доказанные факты и не предлагать опровергнутые решения**.

---

## 1. Навигация по документации репозитория

Перед тем как предлагать архитектурные изменения, гипотезы или проводить замеры, агент **обязан** ознакомиться со следующими файлами:

1. **[`CLAUDE.md`](file:///d:/localllamacpp/CLAUDE.md)** — Главный рабочий журнал проекта.
   * Правила работы, инварианты окружения, точные версии бинарников и хэши коммитов.
   * Протокол измерений и журнал реальных запусков.
2. **[`TUNING.md`](file:///d:/localllamacpp/docs/TUNING.md)** — Физическая модель стоимости и история замеров.
   * Вывод формул, расчет таймингов CPU/GPU/DDR4, история проверенных гипотез.
3. **[`MOE-VIABLE-HYPOTHESES.md`](file:///d:/localllamacpp/docs/MOE-VIABLE-HYPOTHESES.md)** — Реестр подтвержденных и перспективных гипотез ускорения MoE (без сжатия модели).
   * 10 рабочих направлений: *Dynamic Top-$p$ Tail Skipping*, *Dynamic MoE Layer Skipping*, *VRAM HOT/STAGE Cache ($i \to i+2$)*, *Buddy Expert Substitution*, *Layer-Centric Prefill*, *Decoupled Pre-Attention Routing*, *Dual-Batch Interleaving*, и др.
   * Расчеты пропускной способности, формулы, ссылки на научные статьи (2024–2026) и пошаговый Roadmap внедрения.
4. **[`MOE-REJECTED-HYPOTHESES.md`](file:///d:/localllamacpp/docs/MOE-REJECTED-HYPOTHESES.md)** — Реестр отклоненных гипотез с доказательствами (Post-Mortem & Roofline Limits).
   * 12 тупиковых направлений, проверенных и отброшенных:
     - Почему наивный спекулятивный декодинг на DDR4 (MTP / EAGLE-2) замедляет инференс на $-38\%$ (доказательство *Expert Scattering*: 4 токена активируют 30.5 экспертов $\implies$ чтение 2.32 ГБ из DDR4 за шаг).
     - Почему послойный конвейер при Batch=1 физически невозможен (строгий Data Hazard графа Transformer).
     - Почему кастомные CPU GEMV ядра не ускорят модель на Ryzen 5 5600H (DDR4 шина уже на $100\%$ насыщена на скорости $38.5\text{ ГБ/с}$).
     - Почему смещение редукции на CPU (`moe-src-locality.patch`) замедляет TG на $-2.3\%$.
     - Почему Fiddler, полный DMA-стриминг и автономный BuddyMoE при 7% VRAM не работают на этой конфигурации.
5. **[`IGPU-PLAN.md`](file:///d:/localllamacpp/docs/IGPU-PLAN.md)** — План и анализ использования встроенной графики (Vega 7).
6. **[`VISION-BENCH-PLAN.md`](file:///d:/localllamacpp/docs/VISION-BENCH-PLAN.md)** — Планы тестирования Vision-моделей.

Раскладка папок (`docs/`, `bench/`, `vision/`, `tools/`, `prompts/`, `alice/`, `legacy/`) — в разделе «Раскладка репозитория» в `CLAUDE.md`. Строки запуска в журнале сняты до переезда скриптов в папки.

---

## 2. Аппаратный инвариант системы

* **GPU:** NVIDIA GeForce RTX 3060 Laptop, 6144 MiB GDDR6 (192-bit, $\sim 300\text{ ГБ/с}$, PCIe Gen 3.0 x8 $\sim 6.4\text{ ГБ/с}$).
* **CPU:** AMD Ryzen 5 5600H (Zen 3, 6 ядер / 12 потоков, AVX2, **без AVX-512 и AMX**).
* **RAM:** 32 GB DDR4-3200 (Dual-channel, практический пик memcpy $\approx 38.0\text{ ГБ/с}$, gather $\approx 26.5\text{ ГБ/с}$).
* **Основная модель:** `models/Tiel-Coder-35B-A3B-MTP/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf` (40 слоев MoE, 256 экспертов, $\text{top-}k = 8$, гибрид SSM + Attention).

---

## 3. Ключевые правила для агентов

1. **Не предлагать решения из [`MOE-REJECTED-HYPOTHESES.md`](file:///d:/localllamacpp/docs/MOE-REJECTED-HYPOTHESES.md).** Они уже детально исследованы и математически опровергнуты.
2. **Ничего не утверждать без замера.** Расчет по формулам — это гипотеза, замер — факт.
3. **Фиксировать каждый замер в [`CLAUDE.md`](file:///d:/localllamacpp/CLAUDE.md)** с полной строкой запуска и хэшем коммита.
