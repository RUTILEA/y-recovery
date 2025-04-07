#!/bin/bash
# Run Python scripts in the background without -it flag
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data1 /workspace/data/results/OK_data1 0 &
PID1=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data2 /workspace/data/results/OK_data1 0 &
PID2=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data3 /workspace/data/results/OK_data3 0 &
PID3=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data4 /workspace/data/results/OK_data4 3 &
PID4=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data5 /workspace/data/results/OK_data5 3 &
PID5=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data6 /workspace/data/results/OK_data6 3 &
PID6=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data7 /workspace/data/results/OK_data7 4 &
PID7=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data8 /workspace/data/results/OK_data8 4 &
PID8=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data9 /workspace/data/results/OK_data9 4 &
PID9=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data10 /workspace/data/results/OK_data10 5 &
PID10=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data11 /workspace/data/results/OK_data11 5 &
PID11=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data12 /workspace/data/results/OK_data12 5 &
PID12=$!
docker exec yuasa-server python3 inference_main.py /workspace/data/OK_data13 /workspace/data/results/OK_data13 0 &
PID13=$!

# Wait for all processes to complete
echo "Running all Python scripts..."
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9 $PID10 $PID11 $PID12 $PID13
echo "All Python scripts have completed."
exit 0