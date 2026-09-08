"""Compact UI numbers without quantizing unchanged calibration parameters."""


def set_numeric_field(field, value):
    value = float(value)
    rendered = f'{value:.4f}'.rstrip('0').rstrip('.')
    if rendered == '-0':
        rendered = '0'
    field.setProperty('calibration_number', value)
    field.setProperty('calibration_rendered', rendered)
    field.setText(rendered)


def read_numeric_field(field):
    text = field.text().strip()
    if text == field.property('calibration_rendered'):
        return float(field.property('calibration_number'))
    return float(text)
