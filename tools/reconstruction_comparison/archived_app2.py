"""Import an existing native APP2 tree without retracing or moving its seed."""
from pathlib import Path
import math

from .metrics import read_swc

MARKER = '# archived_app2_native_to_image_zero_based_v1'


def convert(source, destination, height):
    """Reflect only Y: native zero-based Y -> image zero-based H-1-Y.

    Preserve node IDs, types, X/Z, radii, parents and all tree components.
    Marker-file +2 compensation does not apply to existing SWC coordinates.
    """
    source = Path(source);destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Archived input must never be overwritten')
    lines = source.read_text().splitlines()
    if any('app2_native_to_image' in s or 'y_axis_flipped_to_image' in s for s in lines):
        raise ValueError('Input already has converted coordinates; refusing another Y flip')
    if not isinstance(height, int) or height <= 0:
        raise ValueError('Image height must be a positive integer')
    read_swc(source)  # Reject empty files before writing anything.
    converted = [MARKER]
    for line in lines:
        fields = line.split()
        if not fields or line.lstrip().startswith('#'):
            converted.append(line);continue
        if len(fields) < 7:
            raise ValueError('Invalid SWC row: '+line)
        y = float(fields[3])
        if not math.isfinite(y):raise ValueError('Nonfinite SWC Y')
        fields[3] = format(height - 1 - y, '.17g')
        converted.append(' '.join(fields))
    content = '\n'.join(converted) + '\n'
    if destination.exists():
        if destination.read_text() == content:return
        raise ValueError('Conflicting imported APP2 output: '+str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
