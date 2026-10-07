import copy as mod_copy
import enum as mod_enum
from dataclasses import dataclass as mod_dataclass
from typing import Any, Callable, Optional

try:
    import xml.etree.cElementTree as mod_etree
except ImportError:
    import xml.etree.ElementTree as mod_etree  # type: ignore

from . import gpx as mod_gpx
from . import utils as mod_utils


DEFAULT_EXTENSION_NAMESPACE = 'https://gpxpy.org/conversion/1.0'
DEFAULT_EXTENSION_PREFIX = 'gpx10'
MIGRATION_MARKER_NAMESPACE = 'https://gpxpy.org/conversion'
MIGRATED_ATTRIBUTE = f'{{{MIGRATION_MARKER_NAMESPACE}}}gpxpy-migrated-from'


class ConversionItemKind(mod_enum.Enum):
    NATIVE = 'native'
    EXTENSION = 'extension'
    PRESERVED_EXTENSION = 'preserved_extension'
    UNSUPPORTED = 'unsupported'
    CONFLICT = 'conflict'


@mod_dataclass(frozen=True)
class ConversionItem:
    kind: ConversionItemKind
    path: str
    field: str
    source_representation: str
    target_representation: str
    value: Any = None
    namespace: Optional[str] = None


@mod_dataclass(frozen=True)
class ConversionReport:
    source_version: str
    target_version: str
    extension_namespace: str
    items: tuple[ConversionItem, ...]

    @property
    def native_items(self) -> tuple[ConversionItem, ...]:
        return self._items(ConversionItemKind.NATIVE)

    @property
    def extension_items(self) -> tuple[ConversionItem, ...]:
        return self._items(ConversionItemKind.EXTENSION)

    @property
    def preserved_extension_items(self) -> tuple[ConversionItem, ...]:
        return self._items(ConversionItemKind.PRESERVED_EXTENSION)

    @property
    def unsupported_items(self) -> tuple[ConversionItem, ...]:
        return self._items(ConversionItemKind.UNSUPPORTED)

    @property
    def conflict_items(self) -> tuple[ConversionItem, ...]:
        return self._items(ConversionItemKind.CONFLICT)

    def has_losses(self) -> bool:
        return bool(self.unsupported_items or self.conflict_items)

    def _items(self, kind: ConversionItemKind) -> tuple[ConversionItem, ...]:
        return tuple(item for item in self.items if item.kind == kind)


class GPXVersionConversionException(mod_gpx.GPXException):
    def __init__(self, message: str, report: ConversionReport):
        super().__init__(message)
        self.report = report


def convert_version(gpx: mod_gpx.GPX, target_version: str, *, strict: bool=False,
                    extension_namespace: str=DEFAULT_EXTENSION_NAMESPACE,
                    extension_prefix: str=DEFAULT_EXTENSION_PREFIX) -> tuple[mod_gpx.GPX, ConversionReport]:
    """
    Return a converted GPX copy and an item-by-item conversion report.

    With strict=False, values without a native representation in the target
    version are moved to extension elements. strict=True rejects extension
    migration, extension-only data that cannot be represented in GPX 1.0, and
    conflicts with pre-existing same-name extensions.
    """
    if target_version not in ('1.0', '1.1'):
        raise mod_gpx.GPXException(f'Invalid target version: {target_version}')
    source_version = str(gpx.version or '')
    if source_version not in ('1.0', '1.1'):
        raise mod_gpx.GPXException(f'Invalid source version: {source_version}')
    if source_version == target_version:
        raise mod_gpx.GPXException(f'GPX is already version {target_version}')

    converted = mod_copy.deepcopy(gpx)
    converted.version = target_version
    extension_prefix = _resolve_extension_prefix(
        converted.nsmap, extension_namespace, extension_prefix)
    ctx = _Context(source_version, target_version, extension_namespace, extension_prefix)

    if target_version == '1.1':
        _convert_10_to_11(converted, ctx)
    else:
        _convert_11_to_10(converted, ctx)

    items = tuple(ctx.items)
    if strict:
        items = tuple(_strict_item(item, target_version) for item in items)
    report = ConversionReport(str(source_version), target_version, extension_namespace, items)
    if strict and _is_strict_blocker(report):
        raise GPXVersionConversionException(
            f'Strict conversion from GPX {source_version} to {target_version} would lose data', report)
    return converted, report


def _resolve_extension_prefix(nsmap: dict[str, str], namespace: str, preferred: str) -> str:
    for prefix, mapped_namespace in nsmap.items():
        if prefix != 'defaultns' and mapped_namespace == namespace:
            return prefix
    candidate = preferred
    suffix = 2
    while candidate in nsmap:
        candidate = f'{preferred}{suffix}'
        suffix += 1
    nsmap[candidate] = namespace
    return candidate

def _strict_item(item: ConversionItem, target_version: str) -> ConversionItem:
    if item.kind == ConversionItemKind.EXTENSION:
        return ConversionItem(ConversionItemKind.UNSUPPORTED, item.path, item.field,
                              item.source_representation, 'not represented', item.value, None)
    if target_version == '1.0' and item.kind == ConversionItemKind.PRESERVED_EXTENSION:
        return ConversionItem(ConversionItemKind.UNSUPPORTED, item.path, item.field,
                              item.source_representation, 'not represented', item.value, None)
    return item


def _is_strict_blocker(report: ConversionReport) -> bool:
    if report.unsupported_items or report.conflict_items:
        return True
    if report.target_version == '1.0':
        return bool(report.extension_items or report.preserved_extension_items)
    return bool(report.extension_items)


class _Context:
    def __init__(self, source_version: str, target_version: str, namespace: str, prefix: str) -> None:
        self.source_version = source_version
        self.target_version = target_version
        self.namespace = namespace
        self.prefix = prefix
        self.items: list[ConversionItem] = []

    def qname(self, tag: str) -> str:
        return f'{{{self.namespace}}}{tag}'

    def native(self, path: str, field: str, source: str, target: str, value: Any) -> None:
        self._add(ConversionItemKind.NATIVE, path, field, source, target, value, None)

    def extension(self, path: str, field: str, source: str, target: str, value: Any) -> None:
        self._add(ConversionItemKind.EXTENSION, path, field, source, target, value, self.namespace)

    def preserved(self, path: str, field: str, element: Any) -> None:
        namespace, _, localname = element.tag.partition('}')
        namespace = namespace[1:] if namespace.startswith('{') else None
        representation = f'extension:{element.tag}'
        self._add(ConversionItemKind.PRESERVED_EXTENSION, path, localname or field,
                  representation, representation, element, namespace)

    def unsupported(self, path: str, field: str, source: str, value: Any) -> None:
        self._add(ConversionItemKind.UNSUPPORTED, path, field, source, 'not represented', value, None)

    def conflict(self, path: str, field: str, source: str, target: str, value: Any) -> None:
        self._add(ConversionItemKind.CONFLICT, path, field, source, target, value, self.namespace)

    def _add(self, kind: ConversionItemKind, path: str, field: str, source: str,
             target: str, value: Any, namespace: Optional[str]) -> None:
        self.items.append(ConversionItem(kind, path, field, source, target, value, namespace))


def _convert_10_to_11(gpx: mod_gpx.GPX, ctx: _Context) -> None:
    _decode_root_10_extensions(gpx, '/gpx', ctx)
    _report_root_native_10_to_11(gpx, '/gpx', ctx)
    _report_preserved(gpx.extensions, '/gpx/extensions', ctx)
    _report_preserved(gpx.metadata_extensions, '/gpx/metadata/extensions', ctx)

    for index, waypoint in enumerate(gpx.waypoints):
        path = f'/gpx/wpt[{index}]'
        _report_common_links(waypoint, path, '1.0', ctx)
        _decode_point_10_extensions(waypoint, path, ctx)
        _report_preserved(waypoint.extensions, f'{path}/extensions', ctx)

    for route_index, route in enumerate(gpx.routes):
        path = f'/gpx/rte[{route_index}]'
        _report_common_links(route, path, '1.0', ctx)
        _decode_container_10_extensions(route, path, ctx)
        _report_preserved(route.extensions, f'{path}/extensions', ctx)
        for point_index, point in enumerate(route.points):
            point_path = f'{path}/rtept[{point_index}]'
            _report_common_links(point, point_path, '1.0', ctx)
            _decode_point_10_extensions(point, point_path, ctx)
            _report_preserved(point.extensions, f'{point_path}/extensions', ctx)

    for track_index, track in enumerate(gpx.tracks):
        path = f'/gpx/trk[{track_index}]'
        _report_common_links(track, path, '1.0', ctx)
        _decode_container_10_extensions(track, path, ctx)
        _report_preserved(track.extensions, f'{path}/extensions', ctx)
        for segment_index, segment in enumerate(track.segments):
            segment_path = f'{path}/trkseg[{segment_index}]'
            _report_preserved(segment.extensions, f'{segment_path}/extensions', ctx)
            for point_index, point in enumerate(segment.points):
                point_path = f'{segment_path}/trkpt[{point_index}]'
                _report_common_links(point, point_path, '1.0', ctx)
                _decode_point_10_extensions(point, point_path, ctx)
                _move_track_point_to_11_extensions(point, point_path, ctx)
                _report_preserved(point.extensions, f'{point_path}/extensions', ctx)



def _decode_root_10_extensions(gpx: mod_gpx.GPX, path: str, ctx: _Context) -> None:
    field_tags = {
        'author-link': ('author_link', 'gpx11:metadata/author/link/@href'),
        'author-link-text': ('author_link_text', 'gpx11:metadata/author/link/text'),
        'author-link-type': ('author_link_type', 'gpx11:metadata/author/link/type'),
        'copyright-author': ('copyright_author', 'gpx11:metadata/copyright/@author'),
        'copyright-year': ('copyright_year', 'gpx11:metadata/copyright/year'),
        'copyright-license': ('copyright_license', 'gpx11:metadata/copyright/license'),
        'link-type': ('link_type', 'gpx11:metadata/link/type'),
    }

    wrapper = _existing(gpx.extensions, ctx.qname('metadata-11'))
    if wrapper is not None and wrapper.get(MIGRATED_ATTRIBUTE) == '1.1':
        for tag, (field, target) in field_tags.items():
            direct_same = [element for element in gpx.extensions if element.tag == ctx.qname(tag)]
            wrapper_same = [child for child in wrapper if child.tag == ctx.qname(tag)]
            migrated_direct = [element for element in direct_same
                               if element.get(MIGRATED_ATTRIBUTE) == '1.1']
            migrated_wrapper = [element for element in wrapper_same
                                if element.get(MIGRATED_ATTRIBUTE) == '1.1']
            migrated = migrated_direct + migrated_wrapper
            if not migrated:
                continue
            selected = migrated[0]
            same_tag = direct_same + wrapper_same
            for element in migrated[1:]:
                ctx.conflict(f'{path}/metadata', field, f'{ctx.prefix}:{tag}', target, element.text)
            for element in same_tag:
                if element.get(MIGRATED_ATTRIBUTE) != '1.1':
                    ctx.conflict(f'{path}/metadata', field, f'{ctx.prefix}:{tag}', target, element.text)
            if getattr(gpx, field) is not None:
                ctx.conflict(f'{path}/metadata', field, f'{ctx.prefix}:{tag}', target, selected.text)
            else:
                setattr(gpx, field, selected.text)
                ctx.native(f'{path}/metadata', field, f'{ctx.prefix}:{tag}', target, selected.text)
                if selected in wrapper:
                    wrapper.remove(selected)
                else:
                    gpx.extensions.remove(selected)

        for child in list(wrapper):
            gpx.metadata_extensions.append(mod_copy.deepcopy(child))
            ctx.preserved(f'{path}/metadata/extensions', child.tag, child)
        gpx.extensions.remove(wrapper)
        ctx.native(f'{path}/metadata', 'metadata_extensions',
                   f'{ctx.prefix}:metadata-11', 'gpx11:metadata/extensions', wrapper)

    for tag, (field, target) in field_tags.items():
        direct_same = [element for element in list(gpx.extensions) if element.tag == ctx.qname(tag)]
        migrated = [element for element in direct_same if element.get(MIGRATED_ATTRIBUTE) == '1.1']
        if not migrated:
            continue
        selected = migrated[0]
        for element in migrated[1:]:
            ctx.conflict(f'{path}/metadata', field, f'{ctx.prefix}:{tag}', target, element.text)
        for element in direct_same:
            if element.get(MIGRATED_ATTRIBUTE) != '1.1':
                ctx.conflict(f'{path}/metadata', field, f'{ctx.prefix}:{tag}', target, element.text)
        if getattr(gpx, field) is not None:
            ctx.conflict(f'{path}/metadata', field, f'{ctx.prefix}:{tag}', target, selected.text)
        else:
            setattr(gpx, field, selected.text)
            ctx.native(f'{path}/metadata', field, f'{ctx.prefix}:{tag}', target, selected.text)
            gpx.extensions.remove(selected)


def _convert_11_to_10(gpx: mod_gpx.GPX, ctx: _Context) -> None:
    _move_root_fields_to_extensions(gpx, '/gpx', ctx)
    _report_root_native_11_to_10(gpx, '/gpx', ctx)
    _wrap_metadata_extensions(gpx, ctx)
    _report_preserved(gpx.extensions, '/gpx/extensions', ctx)
    _report_preserved(gpx.metadata_extensions, '/gpx/metadata/extensions', ctx)

    for index, waypoint in enumerate(gpx.waypoints):
        path = f'/gpx/wpt[{index}]'
        _move_point_link_type(waypoint, path, ctx)
        _report_common_links(waypoint, path, '1.1', ctx)
        _report_preserved(waypoint.extensions, f'{path}/extensions', ctx)

    for route_index, route in enumerate(gpx.routes):
        path = f'/gpx/rte[{route_index}]'
        _move_container_field(route, path, 'type', 'type', 'gpx11:type', ctx)
        _move_point_link_type(route, path, ctx)
        _report_common_links(route, path, '1.1', ctx)
        _report_preserved(route.extensions, f'{path}/extensions', ctx)
        for point_index, point in enumerate(route.points):
            point_path = f'{path}/rtept[{point_index}]'
            _move_point_link_type(point, point_path, ctx)
            _report_common_links(point, point_path, '1.1', ctx)
            _report_preserved(point.extensions, f'{point_path}/extensions', ctx)

    for track_index, track in enumerate(gpx.tracks):
        path = f'/gpx/trk[{track_index}]'
        _move_container_field(track, path, 'type', 'type', 'gpx11:type', ctx)
        _move_point_link_type(track, path, ctx)
        _report_common_links(track, path, '1.1', ctx)
        _report_preserved(track.extensions, f'{path}/extensions', ctx)
        for segment_index, segment in enumerate(track.segments):
            segment_path = f'{path}/trkseg[{segment_index}]'
            _report_preserved(segment.extensions, f'{segment_path}/extensions', ctx)
            for point_index, point in enumerate(segment.points):
                point_path = f'{segment_path}/trkpt[{point_index}]'
                _move_point_link_type(point, point_path, ctx)
                _decode_track_point_to_10(point, point_path, ctx)
                _report_common_links(point, point_path, '1.1', ctx)
                _report_preserved(point.extensions, f'{point_path}/extensions', ctx)

def _value_text(value: Any) -> str:
    return mod_utils.make_str(value)


def _is_conversion_element(element: Any, ctx: _Context, tag: Optional[str]=None) -> bool:
    if not element.tag.startswith(f'{{{ctx.namespace}}}'):
        return False
    if element.get(MIGRATED_ATTRIBUTE) is None:
        return False
    if tag is None:
        return True
    return element.tag == ctx.qname(tag)


def _existing(extensions: list[Any], tag: str) -> Optional[Any]:
    for element in extensions:
        if element.tag == tag:
            return element
    return None


def _migration_element(ctx: _Context, tag: str, value: Any) -> Any:
    element = mod_etree.Element(ctx.qname(tag))
    element.text = _value_text(value)
    element.set(MIGRATED_ATTRIBUTE, ctx.source_version)
    return element


def _move_to_extensions(obj: Any, extensions: list[Any], path: str, field: str,
                        tag: str, source: str, ctx: _Context) -> bool:
    value = getattr(obj, field)
    if value is None:
        return True
    target_tag = ctx.qname(tag)
    same_tag = [element for element in extensions if element.tag == target_tag]
    for element in same_tag:
        ctx.conflict(path, field, source, f'{ctx.prefix}:{tag}', element.text)
    ctx.extension(path, field, source, f'{ctx.prefix}:{tag}', value)
    extensions.append(_migration_element(ctx, tag, value))
    setattr(obj, field, None)
    return True


def _report_preserved(extensions: list[Any], path: str, ctx: _Context) -> None:
    for index, element in enumerate(extensions):
        if _is_conversion_element(element, ctx):
            continue
        ctx.preserved(f'{path}[{index}]', element.tag, element)


def _report_common_links(obj: Any, path: str, source_version: str, ctx: _Context) -> None:
    if source_version == '1.0':
        link_source, link_target = 'gpx10:url', 'gpx11:link/@href'
        text_source, text_target = 'gpx10:urlname', 'gpx11:link/text'
    else:
        link_source, link_target = 'gpx11:link/@href', 'gpx10:url'
        text_source, text_target = 'gpx11:link/text', 'gpx10:urlname'
    if obj.link is not None:
        ctx.native(f'{path}/link', 'link', link_source, link_target, obj.link)
    if obj.link_text is not None:
        ctx.native(f'{path}/link', 'link_text', text_source, text_target, obj.link_text)
    if source_version == '1.1' and obj.link_type is not None:
        # Moved by _move_point_link_type before this report; any remaining value conflicts.
        ctx.unsupported(f'{path}/link', 'link_type', 'gpx11:link/type', obj.link_type)


def _move_track_point_to_11_extensions(point: Any, path: str, ctx: _Context) -> None:
    for field in ('speed', 'course'):
        value = getattr(point, field)
        if value is None:
            continue
        tag = ctx.qname(field)
        same_tag = [element for element in point.extensions if element.tag == tag]
        for element in same_tag:
            ctx.conflict(path, field, f'gpx10:{field}', f'{ctx.prefix}:{field}', element.text)
        ctx.extension(path, field, f'gpx10:{field}', f'{ctx.prefix}:{field}', value)
        point.extensions.append(_migration_element(ctx, field, value))
        setattr(point, field, None)


def _decode_track_point_to_10(point: Any, path: str, ctx: _Context) -> None:
    for field in ('speed', 'course'):
        _restore_marked_scalar(point, point.extensions, path, field, field,
                               f'gpx10:{field}', ctx, '1.0', float)


def _decode_point_10_extensions(obj: Any, path: str, ctx: _Context) -> None:
    _decode_migrated_scalar(obj, obj.extensions, path, 'link_type', 'link-type',
                            'gpx11:link/type', ctx)


def _decode_container_10_extensions(obj: Any, path: str, ctx: _Context) -> None:
    _decode_migrated_scalar(obj, obj.extensions, path, 'link_type', 'link-type',
                            'gpx11:link/type', ctx)
    _decode_migrated_scalar(obj, obj.extensions, path, 'type', 'type', 'gpx11:type', ctx)


def _decode_migrated_scalar(obj: Any, extensions: list[Any], path: str, field: str,
                            tag: str, target: str, ctx: _Context) -> None:
    _restore_marked_scalar(obj, extensions, path, field, tag, target, ctx, '1.1', str)


def _restore_marked_scalar(obj: Any, extensions: list[Any], path: str, field: str,
                           tag: str, target: str, ctx: _Context, marker_version: str,
                           converter: Callable[[str], Any]) -> None:
    target_tag = ctx.qname(tag)
    same_tag = [element for element in extensions if element.tag == target_tag]
    migrated = [element for element in same_tag
                if element.get(MIGRATED_ATTRIBUTE) == marker_version and element.text is not None]
    if not migrated:
        return
    selected = migrated[0]
    selected_text = selected.text
    assert selected_text is not None
    for element in migrated[1:]:
        value = converter(element.text) if element.text is not None else None
        ctx.conflict(path, field, f'{ctx.prefix}:{tag}', target, value)
    for element in same_tag:
        if element.get(MIGRATED_ATTRIBUTE) != marker_version:
            value = converter(element.text) if element.text is not None else None
            ctx.conflict(path, field, f'{ctx.prefix}:{tag}', target, value)
    if getattr(obj, field) is not None:
        ctx.conflict(path, field, f'{ctx.prefix}:{tag}', target, converter(selected_text))
        return
    value = converter(selected_text)
    setattr(obj, field, value)
    ctx.native(path, field, f'{ctx.prefix}:{tag}', target, value)
    extensions.remove(selected)


def _move_point_link_type(obj: Any, path: str, ctx: _Context) -> None:
    _move_to_extensions(obj, obj.extensions, f'{path}/link', 'link_type', 'link-type',
                        'gpx11:link/type', ctx)


def _move_container_field(obj: Any, path: str, field: str, tag: str, source: str, ctx: _Context) -> None:
    _move_to_extensions(obj, obj.extensions, path, field, tag, source, ctx)


def _report_root_native_10_to_11(gpx: mod_gpx.GPX, path: str, ctx: _Context) -> None:
    _report_common_links(gpx, path, '1.0', ctx)
    if gpx.author_name is not None:
        ctx.native(f'{path}/metadata/author', 'author_name', 'gpx10:author',
                   'gpx11:metadata/author/name', gpx.author_name)
    if gpx.author_email is not None:
        ctx.native(f'{path}/metadata/author', 'author_email', 'gpx10:email',
                   'gpx11:metadata/author/email/@id+@domain', gpx.author_email)
    _report_simple_root_metadata(gpx, path, ctx)


def _report_root_native_11_to_10(gpx: mod_gpx.GPX, path: str, ctx: _Context) -> None:
    _report_common_links(gpx, path, '1.1', ctx)
    if gpx.author_name is not None:
        ctx.native(f'{path}/author', 'author_name', 'gpx11:metadata/author/name',
                   'gpx10:author', gpx.author_name)
    if gpx.author_email is not None:
        ctx.native(f'{path}/author', 'author_email', 'gpx11:metadata/author/email/@id+@domain',
                   'gpx10:email', gpx.author_email)
    _report_simple_root_metadata(gpx, path, ctx)


def _report_simple_root_metadata(gpx: mod_gpx.GPX, path: str, ctx: _Context) -> None:
    if ctx.target_version == '1.1':
        mappings = (
            ('name', 'gpx10:name', 'gpx11:metadata/name'),
            ('description', 'gpx10:desc', 'gpx11:metadata/desc'),
            ('time', 'gpx10:time', 'gpx11:metadata/time'),
            ('keywords', 'gpx10:keywords', 'gpx11:metadata/keywords'),
            ('bounds', 'gpx10:bounds', 'gpx11:metadata/bounds'),
        )
    else:
        mappings = (
            ('name', 'gpx11:metadata/name', 'gpx10:name'),
            ('description', 'gpx11:metadata/desc', 'gpx10:desc'),
            ('time', 'gpx11:metadata/time', 'gpx10:time'),
            ('keywords', 'gpx11:metadata/keywords', 'gpx10:keywords'),
            ('bounds', 'gpx11:metadata/bounds', 'gpx10:bounds'),
        )
    for field, source, target in mappings:
        value = getattr(gpx, field)
        if value is not None:
            ctx.native(f'{path}/metadata', field, source, target, value)


def _move_root_fields_to_extensions(gpx: mod_gpx.GPX, path: str, ctx: _Context) -> None:
    root_fields = (
        ('author_link', 'author-link', 'gpx11:metadata/author/link/@href'),
        ('author_link_text', 'author-link-text', 'gpx11:metadata/author/link/text'),
        ('author_link_type', 'author-link-type', 'gpx11:metadata/author/link/type'),
        ('copyright_author', 'copyright-author', 'gpx11:metadata/copyright/@author'),
        ('copyright_year', 'copyright-year', 'gpx11:metadata/copyright/year'),
        ('copyright_license', 'copyright-license', 'gpx11:metadata/copyright/license'),
        ('link_type', 'link-type', 'gpx11:metadata/link/type'),
    )
    for field, tag, source in root_fields:
        _move_to_extensions(gpx, gpx.extensions, f'{path}/metadata', field, tag, source, ctx)


def _wrap_metadata_extensions(gpx: mod_gpx.GPX, ctx: _Context) -> None:
    if not gpx.metadata_extensions:
        return
    wrapper_tag = ctx.qname('metadata-11')
    if _existing(gpx.extensions, wrapper_tag) is not None:
        ctx.conflict('/gpx/metadata/extensions', 'metadata_extensions',
                    'gpx11:metadata/extensions', f'{ctx.prefix}:metadata-11',
                    gpx.metadata_extensions)
        return
    wrapper = mod_etree.Element(wrapper_tag)
    wrapper.set(MIGRATED_ATTRIBUTE, '1.1')
    for element in gpx.metadata_extensions:
        wrapper.append(mod_copy.deepcopy(element))
        ctx.preserved('/gpx/metadata/extensions', element.tag, element)
    gpx.metadata_extensions = []
    gpx.extensions.insert(0, wrapper)
    ctx.extension('/gpx/metadata', 'metadata_extensions', 'gpx11:metadata/extensions',
                  f'{ctx.prefix}:metadata-11', wrapper)
