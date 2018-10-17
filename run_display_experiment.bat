:: call activate vrlatency3

:: for 120 Hz 1200 sample seems fine. "fine" means it covers the transition.
call measure_latency display --port COM11 --trials 2000 --jitter --nsamples 1000 --stimsize 700 --screen 1 --singlemode --output \\THETA\storage\nickdg\vrlatency
::call measure_latency display --port COM11 --trials 10 --jitter --nsamples 1900 --stimsize 700 --screen 1 --singlemode --output \\THETA\storage\nickdg\vrlatency --delay .5

:: for 60 Hz 1800 sample seems fine. "fine" means it covers the transition.
:: call measure_latency display --port COM11   --trials 50 --jitter --nsamples 1800 --stimsize 1000 --screen 1 --singlemode --output \\THETA\storage\nickdg\vrlatency