:: call activate vrlatency3

:: for 120 Hz 1200 sample seems fine. "fine" means it covers the transition.
call measure_latency total --port COM11 --trials 1000 --jitter --nsamples 1100 --stimsize 7 --stimdistance -1.5 --screen 1 --singlemode --output \\THETA\storage\nickdg\vrlatency
