### Create `example_config.yaml` file
```yaml
result_path: "./example_result/"
labels: ["two", "twenty"]
```


### Data augmentation
```bash
python yolo_example.py
```

### Check data
If you run on mac os, you install more lib before check data, otherwise skip this step!
```bash
pip install opencv-python-headless
```
```bash
python yolo_result_gui.py
```
