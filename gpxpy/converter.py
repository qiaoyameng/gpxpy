# Copyright 2011 Tomo Krajina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy as mod_copy
from typing import Any, Callable, NamedTuple, Optional, cast

try:
    import xml.etree.cElementTree as mod_etree
except ImportError:
    import xml.etree.ElementTree as mod_etree  # type: ignore

from . import gpx as mod_gpx
from . import utils as mod_utils

GPX_10_NAMESPACE = 'http://www.topografix.com/GPX/1/0'
GPX_11_NAMESPACE = 'http://www.topografix.com/GPX/1/1'

MODE_STRICT = 'strict'
MODE_ALLOW_EXTENSIONS = 'allow_extensions'

NATIVE_MAPPED = 'native_mapped'
MOVED_TO_EXTENSION = 'moved_to_extension'
UNABLE_TO_PRESERVE = 'unable_to_preserve'
UNKNOWN_PRESERVED = 'unknown_preserved'
EXTENSION_CONFLICT = 'extension_conflict'

CONVERTED_FROM_ATTRIBUTE = 'converted-from'


class ConversionItem(NamedTuple):
    status: str
    field: str
    source_path: str
    destination_path: str
    namespace: Optional[str]
    source_value: Any
    destination_value: Any
    message: str


class ConversionReport(NamedTuple):
    source_version: str
    target_version: str
    mode: str
    items: list[ConversionItem]

    def by_status(self, status: str) -> list[ConversionItem]:
        return [item for item in self.items if item.status == status]

    @property
    def losses(self) -> list[ConversionItem]:
        return self.by_status(UNABLE_TO_PRESERVE) + self.by_status(EXTENSION_CONFLICT)

    @property
    def lossless(self) -> bool:
        return not self.losses


class ConversionResult(NamedTuple):
    gpx: mod_gpx.GPX
    report: ConversionReport


class GPXConversionError(mod_gpx.GPXException):
    def __init__(self, result: ConversionResult) -> None:
        super().__init__('GPX conversion would lose or ambiguously preserve data')
        self.result = result


def convert_version(gpx: mod_gpx.GPX, target_version: str,
                    mode: str=MODE_STRICT) -> ConversionResult:
    if target_version not in ('1.0', '1.1'):
        raise mod_gpx.GPXException(f'Invalid target version {target_version}')
    if mode not in (MODE_STRICT, MODE_ALLOW_EXTENSIONS):
        raise mod_gpx.GPXException(f'Invalid conversion mode {mode}')

    source_version = gpx.version or '1.0'
    if source_version not in ('1.0', '1.1'):
        raise mod_gpx.GPXException(f'Invalid source version {source_version}')
    if source_version == target_version:
        report = ConversionReport(source_version, target_version, mode, [])
        return ConversionResult(gpx.clone(), report)

    result_gpx = gpx.clone()
    result_gpx.version = target_version
    result_gpx.schema_locations = []
    items: list[ConversionItem] = []
    context = _ConversionContext(source_version, target_version, mode, items)

    if target_version == '1.1':
        _convert_10_to_11(result_gpx, context)
    else:
        _convert_11_to_10(result_gpx, context)

    report = ConversionReport(source_version, target_version, mode, items)
    result = ConversionResult(result_gpx, report)
    if mode == MODE_STRICT and not report.lossless:
        raise GPXConversionError(result)
    return result


class _ConversionContext:
    def __init__(self, source_version: str, target_version: str,
                 mode: str, items: list[ConversionItem]) -> None:
        self.source_version = source_version
        self.target_version = target_version
        self.mode = mode
        self.items = items
        self.source_namespace = GPX_10_NAMESPACE if source_version == '1.0' else GPX_11_NAMESPACE
        self.target_namespace = GPX_11_NAMESPACE if target_version == '1.1' else GPX_10_NAMESPACE
        self.source_prefix = 'gpx10' if source_version == '1.0' else 'gpx11'

    def native(self, field: str, source_path: str, destination_path: str,
               source_value: Any, destination_value: Any,
               message: str='Mapped to native GPX field') -> None:
        self.items.append(ConversionItem(
            NATIVE_MAPPED, field, source_path, destination_path,
            None, source_value, destination_value, message))

    def moved(self, field: str, source_path: str, destination_path: str,
              namespace: str, source_value: Any, destination_value: Any,
              message: str='Moved to designated extension namespace') -> None:
        self.items.append(ConversionItem(
            MOVED_TO_EXTENSION, field, source_path, destination_path,
            namespace, source_value, destination_value, message))

    def unknown(self, source_path: str, destination_path: str,
                namespace: str, element: mod_etree.Element) -> None:
        self.items.append(ConversionItem(
            UNKNOWN_PRESERVED, _element_field(element), source_path,
            destination_path, namespace, element, element,
            'Unknown extension retained as an XML element'))

    def lost(self, field: str, source_path: str, source_value: Any,
             message: str) -> None:
        self.items.append(ConversionItem(
            UNABLE_TO_PRESERVE, field, source_path, '', None,
            source_value, None, message))

    def conflict(self, field: str, source_path: str, destination_path: str,
                 namespace: str, source_value: Any, destination_value: Any,
                 message: str) -> None:
        self.items.append(ConversionItem(
            EXTENSION_CONFLICT, field, source_path, destination_path,
            namespace, source_value, destination_value, message))


def _qname(namespace: str, local_name: str) -> str:
    return f'{{{namespace}}}{local_name}'


def _element_field(element: mod_etree.Element) -> str:
    tag = str(element.tag)
    if '}' in tag:
        return tag.rpartition('}')[2]
    return tag


def _ensure_namespace(gpx: mod_gpx.GPX, namespace: str, preferred_prefix: str) -> None:
    for prefix, uri in gpx.nsmap.items():
        if uri == namespace and prefix != 'defaultns':
            return
    prefix = preferred_prefix
    suffix = 2
    while prefix in gpx.nsmap:
        prefix = f'{preferred_prefix}{suffix}'
        suffix += 1
    gpx.nsmap[prefix] = namespace


def _is_migration_marker(element: mod_etree.Element) -> bool:
    namespace = _namespace(element)
    if namespace not in (GPX_10_NAMESPACE, GPX_11_NAMESPACE):
        return False
    for attribute_name in element.attrib:
        if attribute_name in (
                _qname(GPX_10_NAMESPACE, CONVERTED_FROM_ATTRIBUTE),
                _qname(GPX_11_NAMESPACE, CONVERTED_FROM_ATTRIBUTE)):
            return True
    return False


def _extension_element(context: _ConversionContext, local_name: str,
                       value: Any, marked: bool) -> mod_etree.Element:
    element = mod_etree.Element(_qname(context.source_namespace, local_name))
    element.text = mod_utils.make_str(value)
    if marked:
        element.set(_qname(context.source_namespace, CONVERTED_FROM_ATTRIBUTE), 'native')
    return element


def _find_extensions(extensions: list[Any], local_name: str,
                     namespace: str, marked: Optional[bool]=None) -> list[mod_etree.Element]:
    tag = _qname(namespace, local_name)
    marker = _qname(namespace, CONVERTED_FROM_ATTRIBUTE)
    result: list[mod_etree.Element] = []
    for extension in extensions:
        if extension.tag != tag:
            continue
        has_marker = marker in extension.attrib
        if marked is None or has_marker == marked:
            result.append(cast(mod_etree.Element, extension))
    return result


def _append_migration(context: _ConversionContext, instance: Any,
                      attribute_name: str, extensions: list[Any],
                      local_name: str, source_path: str, destination_path: str,
                      value: Any, position: Optional[int]=None) -> None:
    if getattr(instance, attribute_name) is not None:
        context.conflict(
            local_name, source_path, destination_path,
            context.source_namespace, value, getattr(instance, attribute_name),
            'Target native field already contains a value')
    if context.mode == MODE_STRICT:
        context.lost(local_name, source_path, value,
                     'Strict mode does not allow preserving a native field in an extension')
        return
    existing_extensions = _find_extensions(
        extensions, local_name, context.source_namespace, False)
    if existing_extensions:
        context.conflict(
            local_name, source_path, destination_path,
            context.source_namespace, value,
            existing_extensions[0],
            'A same-name extension already exists; migrated value was retained with a marker')
    migrated_element = _extension_element(context, local_name, value, True)
    if position is None:
        extensions.append(migrated_element)
    else:
        extensions.insert(position, migrated_element)
    context.moved(local_name, source_path, destination_path,
                  context.source_namespace, value, migrated_element)


def _restore_migration(context: _ConversionContext, instance: Any, field: str,
                       extensions: list[Any], source_path: str,
                       destination_path: str,
                       value_type: Callable[..., Any]=str,
                       attribute_name: Optional[str]=None,
                       only_if: Optional[Callable[[Any], bool]]=None) -> bool:
    attribute_name = attribute_name or field
    if only_if is not None and not only_if(instance):
        return False
    nodes = _find_extensions(extensions, field, context.target_namespace, True)
    if not nodes:
        return False
    node = nodes[0]
    if getattr(instance, attribute_name) is not None:
        context.conflict(
            field, source_path, destination_path, context.target_namespace,
            node, getattr(instance, attribute_name),
            'Native field and migrated extension contain values')
        return False
    if len(nodes) > 1:
        context.conflict(
            field, source_path, destination_path, context.target_namespace,
            nodes, None, 'Multiple migrated extensions use the same field name')
        return False
    text = node.text
    if text is None:
        text = ''
    current_value: Any = value_type(text)
    setattr(instance, attribute_name, current_value)
    extensions.remove(node)
    context.native(field, source_path, destination_path, node, current_value,
                   'Restored designated extension to a native GPX field')
    return True


def _copy_unknown_extensions(context: _ConversionContext, source: list[Any],
                             destination: list[Any], source_base: str,
                             destination_base: str, gpx: mod_gpx.GPX,
                             allowed: bool) -> None:
    for extension in list(source):
        namespace = _namespace(extension)
        source_path = f'{source_base}/{_element_field(extension)}'
        destination_path = f'{destination_base}/{_element_field(extension)}'
        if not allowed:
            context.lost(_element_field(extension), source_path, extension,
                         'Strict mode cannot preserve extensions in target GPX version')
            continue
        if source is not destination:
            destination.append(mod_copy.deepcopy(extension))
        if namespace:
            _ensure_namespace(gpx, namespace, _prefix_for_namespace(gpx, namespace))
            if not _is_migration_marker(extension):
                context.unknown(source_path, destination_path, namespace, extension)


def _namespace(element: mod_etree.Element) -> str:
    tag = str(element.tag)
    if tag.startswith('{'):
        return tag[1:tag.index('}')]
    return ''


def _prefix_for_namespace(gpx: mod_gpx.GPX, namespace: str) -> str:
    for prefix, uri in gpx.nsmap.items():
        if uri == namespace:
            return prefix
    return 'ext'


def _report_unmapped_scalar(context: _ConversionContext, instance: Any,
                            attribute_name: str, extensions: list[Any],
                            field: str, source_path: str,
                            destination_path: str, value: Any,
                            position: Optional[int]=None) -> None:
    if value is None:
        return
    _append_migration(context, instance, attribute_name, extensions, field,
                      source_path, destination_path, value, position)


def _link_paths(path: str, up: bool) -> tuple[str, str]:
    if up:
        return f'{path}/url', f'{path}/link/@href'
    return f'{path}/link/@href', f'{path}/url'


def _native_links_up(context: _ConversionContext, instance: Any, path: str) -> None:
    if instance.link is not None:
        source, destination = _link_paths(path, True)
        context.native('link', source, destination, instance.link, instance.link)
        if instance.link_text is not None:
            context.native('link_text', f'{path}/urlname', f'{path}/link/text',
                           instance.link_text, instance.link_text)
    elif instance.link_text is not None:
        _append_migration(context, instance, 'link_text', instance.extensions,
                          'urlname', f'{path}/urlname',
                          f'{path}/extensions/gpx10:urlname', instance.link_text)
        instance.link_text = None


def _restore_links_up(context: _ConversionContext, instance: Any, path: str) -> None:
    _restore_migration(
        context, instance, 'link_type', instance.extensions,
        f'{path}/extensions/gpx11:link_type', f'{path}/link/type',
        attribute_name='link_type',
        only_if=lambda obj: obj.link is not None)


def _restore_point_10_urlname(context: _ConversionContext, instance: Any, path: str) -> None:
    _restore_migration(
        context, instance, 'urlname', instance.extensions,
        f'{path}/extensions/gpx10:urlname', f'{path}/link/text',
        attribute_name='link_text',
        only_if=lambda obj: obj.link is not None)


def _restore_root_11_fields(gpx: mod_gpx.GPX, context: _ConversionContext) -> None:
    fields = ('author_link', 'author_link_text', 'author_link_type',
              'copyright_author', 'copyright_year', 'copyright_license',
              'link_type')
    for field in fields:
        only_if: Optional[Callable[[Any], bool]] = None
        if field == 'link_type':
            only_if = lambda obj: obj.link is not None
        _restore_migration(
            context, gpx, field, gpx.extensions,
            f'extensions/gpx11:{field}', f'metadata/{field.replace("_", "/")}',
            attribute_name=field, only_if=only_if)


def _restore_root_10_urlname(gpx: mod_gpx.GPX, context: _ConversionContext) -> None:
    _restore_migration(
        context, gpx, 'urlname', gpx.metadata_extensions,
        'metadata/extensions/gpx10:urlname', 'metadata/link/text',
        attribute_name='link_text',
        only_if=lambda obj: obj.link is not None)


def _restore_10_urlname(context: _ConversionContext, instance: Any,
                        path: str, destination_path: str) -> None:
    _restore_migration(
        context, instance, 'urlname', instance.extensions,
        f'{path}/extensions/gpx10:urlname', destination_path,
        attribute_name='link_text')


def _native_links_down(context: _ConversionContext, instance: Any, path: str) -> None:
    if instance.link is not None:
        source, destination = _link_paths(path, False)
        context.native('link', source, destination, instance.link, instance.link)
    if instance.link_text is not None:
        context.native('link_text', f'{path}/link/text', f'{path}/urlname',
                       instance.link_text, instance.link_text)
    if instance.link_type is not None:
        _append_migration(context, instance, 'link_type', instance.extensions,
                          'link_type', f'{path}/link/type',
                          f'{path}/extensions/gpx11:link_type', instance.link_type)
        instance.link_type = None


def _convert_10_to_11(gpx: mod_gpx.GPX, context: _ConversionContext) -> None:
    _ensure_namespace(gpx, context.source_namespace, context.source_prefix)

    _restore_root_11_fields(gpx, context)

    if gpx.author_name is not None:
        context.native('author_name', 'author', 'metadata/author/name',
                       gpx.author_name, gpx.author_name)
    if gpx.author_email is not None:
        context.native('author_email', 'email', 'metadata/author/email',
                       gpx.author_email, gpx.author_email)
    if gpx.link is not None:
        context.native('link', 'url', 'metadata/link/@href', gpx.link, gpx.link)
        if gpx.link_text is not None:
            context.native('link_text', 'urlname', 'metadata/link/text',
                           gpx.link_text, gpx.link_text)
    elif gpx.link_text is not None:
        _append_migration(context, gpx, 'link_text', gpx.metadata_extensions,
                          'urlname', 'urlname',
                          'metadata/extensions/gpx10:urlname', gpx.link_text)
        gpx.link_text = None
    else:
        _restore_root_10_urlname(gpx, context)

    _copy_unknown_extensions(context, gpx.extensions, gpx.metadata_extensions,
                             'extensions', 'metadata/extensions', gpx, True)
    gpx.extensions = []

    for waypoint_index, waypoint in enumerate(gpx.waypoints):
        path = f'wpt[{waypoint_index}]'
        _restore_links_up(context, waypoint, path)
        _native_links_up(context, waypoint, path)
        _copy_unknown_extensions(context, waypoint.extensions, waypoint.extensions,
                                 path, path, gpx, True)

    for route_index, route in enumerate(gpx.routes):
        path = f'rte[{route_index}]'
        _restore_migration(context, route, 'type', route.extensions,
                           f'{path}/extensions/gpx11:type', f'{path}/type')
        _restore_links_up(context, route, path)
        _native_links_up(context, route, path)
        _copy_unknown_extensions(context, route.extensions, route.extensions,
                                 path, path, gpx, True)
        for point_index, point in enumerate(route.points):
            point_path = f'{path}/rtept[{point_index}]'
            _restore_point_10_urlname(context, point, point_path)
            _restore_links_up(context, point, point_path)
            _native_links_up(context, point, point_path)
            _copy_unknown_extensions(context, point.extensions, point.extensions,
                                     point_path, point_path, gpx, True)

    for track_index, track in enumerate(gpx.tracks):
        path = f'trk[{track_index}]'
        _restore_migration(context, track, 'type', track.extensions,
                           f'{path}/extensions/gpx11:type', f'{path}/type')
        _restore_links_up(context, track, path)
        _native_links_up(context, track, path)
        _copy_unknown_extensions(context, track.extensions, track.extensions,
                                 path, path, gpx, True)
        for segment_index, segment in enumerate(track.segments):
            segment_path = f'{path}/trkseg[{segment_index}]'
            _copy_unknown_extensions(context, segment.extensions, segment.extensions,
                                     segment_path, segment_path, gpx, True)
            for point_index, point in enumerate(segment.points):
                point_path = f'{segment_path}/trkpt[{point_index}]'
                _restore_point_10_urlname(context, point, point_path)
                _restore_links_up(context, point, point_path)
                _native_links_up(context, point, point_path)
                for field in ('speed', 'course'):
                    value = getattr(point, field)
                    if value is not None:
                        destination = f'{point_path}/extensions/gpx10:{field}'
                        existing = _find_extensions(point.extensions, field,
                                                    context.source_namespace,
                                                    False)
                        marked = _find_extensions(point.extensions, field,
                                                  context.source_namespace, True)
                        if marked and getattr(marked[0], 'text', None) == mod_utils.make_str(value):
                            point.extensions.remove(marked[0])
                            existing = [extension for extension in existing
                                        if extension is not marked[0]]
                        if existing:
                            context.conflict(
                                field, f'{point_path}/{field}', destination,
                                context.source_namespace, value,
                                existing[0],
                                'A same-name extension already exists')
                        if context.mode == MODE_STRICT:
                            context.lost(field, f'{point_path}/{field}', value,
                                         'GPX 1.1 track points do not natively support this field')
                            setattr(point, field, None)
                        else:
                            migrated_element = _extension_element(context, field, value, True)
                            point.extensions.insert(0, migrated_element)
                            context.moved(field, f'{point_path}/{field}', destination,
                                          context.source_namespace, value,
                                          migrated_element)
                            setattr(point, field, None)
                _copy_unknown_extensions(context, point.extensions, point.extensions,
                                         point_path, point_path, gpx, True)


def _convert_11_to_10(gpx: mod_gpx.GPX, context: _ConversionContext) -> None:
    _ensure_namespace(gpx, context.source_namespace, context.source_prefix)
    allow = context.mode == MODE_ALLOW_EXTENSIONS

    for field in ('author_link', 'author_link_text', 'author_link_type',
                  'copyright_author', 'copyright_year', 'copyright_license',
                  'link_type'):
        _restore_migration(
            context, gpx, field, gpx.extensions,
            f'extensions/gpx11:{field}', f'metadata/{field.replace("_", "/")}',
            attribute_name=field,
            only_if=(lambda obj: obj.link is not None) if field == 'link_type' else None)
    _restore_10_urlname(context, gpx, '', 'urlname')

    if gpx.author_name is not None:
        context.native('author_name', 'metadata/author/name', 'author',
                       gpx.author_name, gpx.author_name)
    if gpx.author_email is not None:
        context.native('author_email', 'metadata/author/email', 'email',
                       gpx.author_email, gpx.author_email)
    for field in ('author_link', 'author_link_text', 'author_link_type',
                  'copyright_author', 'copyright_year', 'copyright_license'):
        value = getattr(gpx, field)
        if value is not None:
            _report_unmapped_scalar(
                context, gpx, field, gpx.extensions, field,
                f'metadata/{field.replace("_", "/")}',
                f'extensions/gpx11:{field}', value)
            setattr(gpx, field, None)
    if gpx.link is not None:
        context.native('link', 'metadata/link/@href', 'url', gpx.link, gpx.link)
    if gpx.link_text is not None:
        context.native('link_text', 'metadata/link/text', 'urlname',
                       gpx.link_text, gpx.link_text)
    if gpx.link_type is not None:
        _report_unmapped_scalar(
            context, gpx, 'link_type', gpx.extensions, 'link_type',
            'metadata/link/type', 'extensions/gpx11:link_type', gpx.link_type)
        gpx.link_type = None

    _copy_unknown_extensions(context, gpx.metadata_extensions, gpx.extensions,
                             'metadata/extensions', 'extensions', gpx, allow)
    gpx.metadata_extensions = []

    for waypoint_index, waypoint in enumerate(gpx.waypoints):
        path = f'wpt[{waypoint_index}]'
        _restore_migration(
            context, waypoint, 'link_type', waypoint.extensions,
            f'{path}/extensions/gpx11:link_type', f'{path}/link/type',
            attribute_name='link_type',
            only_if=lambda obj: obj.link is not None)
        _native_links_down(context, waypoint, path)
        _copy_unknown_extensions(context, waypoint.extensions, waypoint.extensions,
                                 path, path, gpx, allow)

    for route_index, route in enumerate(gpx.routes):
        path = f'rte[{route_index}]'
        if not _restore_migration(context, route, 'type', route.extensions,
                                  f'{path}/extensions/gpx11:type', f'{path}/type'):
            if route.type is not None:
                _report_unmapped_scalar(
                    context, route, 'type', route.extensions, 'type',
                    f'{path}/type', f'{path}/extensions/gpx11:type', route.type)
                route.type = None
        _restore_migration(
            context, route, 'link_type', route.extensions,
            f'{path}/extensions/gpx11:link_type', f'{path}/link/type',
            attribute_name='link_type',
            only_if=lambda obj: obj.link is not None)
        _native_links_down(context, route, path)
        _copy_unknown_extensions(context, route.extensions, route.extensions,
                                 path, path, gpx, allow)
        for point_index, point in enumerate(route.points):
            point_path = f'{path}/rtept[{point_index}]'
            _restore_migration(
                context, point, 'link_type', point.extensions,
                f'{point_path}/extensions/gpx11:link_type',
                f'{point_path}/link/type', attribute_name='link_type',
                only_if=lambda obj: obj.link is not None)
            _native_links_down(context, point, point_path)
            _copy_unknown_extensions(context, point.extensions, point.extensions,
                                     point_path, point_path, gpx, allow)

    for track_index, track in enumerate(gpx.tracks):
        path = f'trk[{track_index}]'
        if not _restore_migration(context, track, 'type', track.extensions,
                                  f'{path}/extensions/gpx11:type', f'{path}/type'):
            if track.type is not None:
                _report_unmapped_scalar(
                    context, track, 'type', track.extensions, 'type',
                    f'{path}/type', f'{path}/extensions/gpx11:type', track.type)
                track.type = None
        _restore_migration(
            context, track, 'link_type', track.extensions,
            f'{path}/extensions/gpx11:link_type', f'{path}/link/type',
            attribute_name='link_type',
            only_if=lambda obj: obj.link is not None)
        _native_links_down(context, track, path)
        _copy_unknown_extensions(context, track.extensions, track.extensions,
                                 path, path, gpx, allow)
        for segment_index, segment in enumerate(track.segments):
            segment_path = f'{path}/trkseg[{segment_index}]'
            _copy_unknown_extensions(context, segment.extensions, segment.extensions,
                                     segment_path, segment_path, gpx, allow)
            for point_index, point in enumerate(segment.points):
                point_path = f'{segment_path}/trkpt[{point_index}]'
                _restore_migration(
                    context, point, 'link_type', point.extensions,
                    f'{point_path}/extensions/gpx11:link_type',
                    f'{point_path}/link/type', attribute_name='link_type',
                    only_if=lambda obj: obj.link is not None)
                _native_links_down(context, point, point_path)
                for field in ('course', 'speed'):
                    _restore_migration(
                        context, point, field, point.extensions,
                        f'{point_path}/extensions/gpx11:{field}',
                        f'{point_path}/{field}', value_type=float)
                _copy_unknown_extensions(context, point.extensions, point.extensions,
                                         point_path, point_path, gpx, allow)
