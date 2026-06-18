# Othello Adversarial Search (Minimax & Alpha-Beta)

پروژه شماره ۳ درس هوش مصنوعی — دانشگاه اصفهان

## ساختار
```
game/othello.py          # محیط بازی (بدون تغییر)
agents/random_agent.py   # عامل تصادفی (بدون تغییر)
agents/greedy_agent.py   # عامل حریص (بدون تغییر)
agents/minimax_agent.py  # ✅ پیاده‌سازی‌شده
agents/alphabeta_agent.py# ✅ پیاده‌سازی‌شده
tournament.py            # موتور اجرای بازی (بدون تغییر)
main.py                  # نمایش بازی‌های نمونه (به‌روزرسانی‌شده)
run_experiments.py       # اجرای آزمایش‌ها و مقایسه‌ها
```

## اجرا
```bash
python main.py             # بازی‌های نمونه میان عامل‌ها
python run_experiments.py  # نرخ برد + مقایسه گره/زمان Minimax و Alpha-Beta
```
