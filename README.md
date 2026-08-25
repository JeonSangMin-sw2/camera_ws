  Upper bounds: [40.0, 94.0, 10.0]
  Optimal residuals: [np.float64(-0.14429796439512899), np.float64(-0.4646992049616472), np.float64(0.4865873679074184), np.float64(0.0), np.float64(1.8198081945740795e-08), np.float64(-3.8842192046031694e-08)]
[VALIDATION] LEFT ARM BRACKET SWEEP CIRCLE RESIDUALS:
  * J6 Sweep Radius Err: 0.1443 mm
  * J5 Sweep Radius Err: 0.4647 mm
  * J4 Sweep Radius Err: 0.4866 mm
  [SUCCESS] Circle reconstruction PASSED (Max Residual: 0.4866 mm < 1.0 mm)
Step 1: Moving to Joint Ready Pose...
Step 2: Moving to Cartesian Checking Pose...
Traceback (most recent call last):
  File "/home/nvidia/camera_ws/main_ui.py", line 5799, in on_finished
    head_cfg = get_head_config(self.model)
  File "/home/nvidia/camera_ws/core/calibration_core.py", line 197, in get_head_config
    camera_nominals = load_camera_nominals()
  File "/home/nvidia/camera_ws/core/calibration_core.py", line 128, in load_camera_nominals
    config = yaml.safe_load(f) or {}
  File "/home/nvidia/camera_ws/.venv/lib/python3.10/site-packages/yaml/__init__.py", line 125, in safe_load
    return load(stream, SafeLoader)
  File "/home/nvidia/camera_ws/.venv/lib/python3.10/site-packages/yaml/__init__.py", line 81, in load
    return loader.get_single_data()
  File "/home/nvidia/camera_ws/.venv/lib/python3.10/site-packages/yaml/constructor.py", line 49, in get_single_data
    node = self.get_single_node()
  File "/home/nvidia/camera_ws/.venv/lib/python3.10/site-packages/yaml/composer.py", line 36, in get_single_node
    document = self.compose_document()
  File "/home/nvidia/camera_ws/.venv/lib/python3.10/site-packages/yaml/composer.py", line 55, in compose_document
    node = self.compose_node(None, None)
  File "/home/nvidia/camera_ws/.venv/lib/python3.10/site-packages/yaml/composer.py", line 84, in compose_node
    node = self.compose_mapping_node(anchor)
  File "/home/nvidia/camera_ws/.venv/lib/python3.10/site-packages/yaml/composer.py", line 133, in compose_mapping_node
    item_value = self.compose_node(node, item_key)
  File "/home/nvidia/camera_ws/.venv/lib/python3.10/site-packages/yaml/composer.py", line 84, in compose_node
    node = self.compose_mapping_node(anchor)
  File "/home/nvidia/camera_ws/.venv/lib/python3.10/site-packages/yaml/composer.py", line 127, in compose_mapping_node
    while not self.check_event(MappingEndEvent):
  File "/home/nvidia/camera_ws/.venv/lib/python3.10/site-packages/yaml/parser.py", line 98, in check_event
    self.current_event = self.state()
  File "/home/nvidia/camera_ws/.venv/lib/python3.10/site-packages/yaml/parser.py", line 438, in parse_block_mapping_key
    raise ParserError("while parsing a block mapping", self.marks[-1],
yaml.parser.ParserError: while parsing a block mapping
  in "/home/nvidia/camera_ws/config/setting.yaml", line 33, column 3
expected <block end>, but found '-'
  in "/home/nvidia/camera_ws/config/setting.yaml", line 34, column 3

