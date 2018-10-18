:: call activate vrlatency3

:: for 120 Hz 1200 sample seems fine. "fine" means it covers the transition.
call measure_latency display --port COM11 --trials 50 --jitter --nsamples 1300 --stimsize 700 --screen 1 --singlemode --output \\THETA\storage\nickdg\vrlatency

:: for 60 Hz 1800 sample seems fine. "fine" means it covers the transition.
:: call measure_latency display --port COM11   --trials 50 --jitter --nsamples 1800 --stimsize 1000 --screen 1 --singlemode --output \\THETA\storage\nickdg\vrlatency