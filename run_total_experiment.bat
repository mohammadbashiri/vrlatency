:: call activate vrlatency3

:: for 120 Hz 1200 sample seems fine. "fine" means it covers the transition.
call measure_latency total --port COM11 --trials 200 --jitter --nsamples 1100 --stimsize 5 --stimdistance -2.3 --screen 1 --singlemode --output \\THETA\storage\nickdg\vrlatency
