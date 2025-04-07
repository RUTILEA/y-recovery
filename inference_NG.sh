#!/bin/bash
# Run Python scripts in the background without -it flag
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_A1 /workspace/data/results/NG_data_A1 0 &
PID1=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_A2 /workspace/data/results/NG_data_A2 0 &
PID2=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_A3 /workspace/data/results/NG_data_A3 0 &
PID3=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_A4 /workspace/data/results/NG_data_A4 3 &
PID4=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_A5 /workspace/data/results/NG_data_A5 3 &
PID5=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_A6 /workspace/data/results/NG_data_A6 3 &
PID6=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_B1 /workspace/data/results/NG_data_B1 4 &
PID7=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_B2 /workspace/data/results/NG_data_B2 4 &
PID8=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_B3 /workspace/data/results/NG_data_B3 4 &
PID9=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_B4 /workspace/data/results/NG_data_B4 5 &
PID10=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_B5 /workspace/data/results/NG_data_B5 5 &
PID11=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/NG_data_B6 /workspace/data/results/NG_data_B6 5 &
PID12=$!

# Wait for all processes to complete
echo "Running all Python scripts..."
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9 $PID10 $PID11 $PID12
echo "All Python scripts have completed."
exit 0