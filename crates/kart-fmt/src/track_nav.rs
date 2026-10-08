//! Discovery parser for the navigation-record vectors embedded in `track.1s`.
//!
//! The legacy track container appends road-query data after its KartObject
//! scene.  Each path vector is encoded as a little-endian `u32` count followed
//! by `count` 36-byte records (`position`, `secondary`, `normal_or_side`).
//! There is no enclosing length available to the scene parser, so discovery
//! validates the physical invariants of every complete vector rather than
//! accepting arbitrary float runs.

use crate::bml::Tag;
use crate::kart_object::Vec3;

#[derive(Clone, Debug, PartialEq)]
pub struct NavRecord {
    pub position: Vec3,
    pub secondary: Vec3,
    pub normal_or_side: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NavRecordVector {
    /// Byte offset of the little-endian count field.
    pub offset: usize,
    pub records: Vec<NavRecord>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToRoadElement {
    pub marker: String,
    pub boundary: Vec<Vec3>,
    pub triangles: Vec<[u16; 3]>,
    pub surface: String,
    pub surface_triangles: Vec<[u16; 3]>,
    pub records: Vec<NavRecord>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToRoad {
    pub offset: usize,
    pub name: String,
    pub flags: u16,
    pub elements: Vec<ToRoadElement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedNavVector {
    pub road: String,
    pub element: usize,
    pub records: Vec<NavRecord>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssembledCourse {
    pub contexts: Vec<SelectedNavVector>,
    pub successors: Vec<Vec<usize>>,
}

pub const TO_ROAD_STAMP: u32 = 0x07DE_0249;
const RECORD_SIZE: usize = 36;

fn read_f32(data: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn norm(vector: Vec3) -> f32 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn decode_record(data: &[u8], offset: usize) -> NavRecord {
    let vector = |base| {
        [
            read_f32(data, offset + base),
            read_f32(data, offset + base + 4),
            read_f32(data, offset + base + 8),
        ]
    };
    NavRecord {
        position: vector(0),
        secondary: vector(12),
        normal_or_side: vector(24),
    }
}

fn physically_valid(record: &NavRecord) -> bool {
    let values = record
        .position
        .into_iter()
        .chain(record.secondary)
        .chain(record.normal_or_side);
    values.clone().all(f32::is_finite)
        && values.map(f32::abs).all(|value| value < 100_000.0)
        && (norm(record.secondary) - 1.0).abs() <= 1.0e-3
        && (norm(record.normal_or_side) - 1.0).abs() <= 1.0e-3
        && dot(record.secondary, record.normal_or_side).abs() <= 2.0e-3
}

struct Cursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, size: usize) -> Option<&'a [u8]> {
        let end = self.offset.checked_add(size)?;
        let bytes = self.data.get(self.offset..end)?;
        self.offset = end;
        Some(bytes)
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn vec3(&mut self) -> Option<Vec3> {
        Some([
            f32::from_le_bytes(self.take(4)?.try_into().ok()?),
            f32::from_le_bytes(self.take(4)?.try_into().ok()?),
            f32::from_le_bytes(self.take(4)?.try_into().ok()?),
        ])
    }

    fn text(&mut self) -> Option<String> {
        let count = self.u32()? as usize;
        if count > 4096 {
            return None;
        }
        let bytes = self.take(count.checked_mul(2)?)?;
        let units = bytes
            .chunks_exact(2)
            .map(|value| u16::from_le_bytes([value[0], value[1]]))
            .collect::<Vec<_>>();
        String::from_utf16(&units).ok()
    }

    fn triangles(&mut self) -> Option<Vec<[u16; 3]>> {
        let count = self.u32()? as usize;
        if count > 65_536 {
            return None;
        }
        (0..count)
            .map(|_| Some([self.u16()?, self.u16()?, self.u16()?]))
            .collect()
    }

    fn records(&mut self) -> Option<Vec<NavRecord>> {
        let count = self.u32()? as usize;
        if count > 4096 {
            return None;
        }
        let records = (0..count)
            .map(|_| {
                Some(NavRecord {
                    position: self.vec3()?,
                    secondary: self.vec3()?,
                    normal_or_side: self.vec3()?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        records.iter().all(physically_valid).then_some(records)
    }
}

pub fn parse_to_road(data: &[u8], offset: usize) -> Option<ToRoad> {
    let mut cursor = Cursor { data, offset };
    (cursor.u16()? == 0x47AA).then_some(())?;
    (cursor.u32()? == TO_ROAD_STAMP).then_some(())?;
    cursor.u16()?; // KartObject instance id
    let name = cursor.text()?;
    let flags = cursor.u16()?;
    let element_count = cursor.u32()? as usize;
    if element_count > 4096 {
        return None;
    }
    let elements = (0..element_count)
        .map(|_| {
            let marker = cursor.text()?;
            let boundary_count = cursor.u32()? as usize;
            if boundary_count > 65_536 {
                return None;
            }
            let boundary = (0..boundary_count)
                .map(|_| cursor.vec3())
                .collect::<Option<Vec<_>>>()?;
            let triangles = cursor.triangles()?;
            let surface = cursor.text()?;
            let surface_triangles = cursor.triangles()?;
            let records = cursor.records()?;
            Some(ToRoadElement {
                marker,
                boundary,
                triangles,
                surface,
                surface_triangles,
                records,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ToRoad {
        offset,
        name,
        flags,
        elements,
    })
}

/// Decode every complete `ToRoad` object embedded in the track asset.
///
/// Unlike [`discover_record_vectors`], this retains road ownership, element
/// markers, boundary geometry and triangle tables.  The discovery scanner is
/// kept as an independent cross-check for record vectors whose owner is not yet
/// selected by a course.
pub fn discover_to_roads(data: &[u8]) -> Vec<ToRoad> {
    let mut roads = Vec::new();
    for offset in 0..data.len().saturating_sub(8) {
        if data[offset..offset + 2] != 0x47AAu16.to_le_bytes()
            || data[offset + 2..offset + 6] != TO_ROAD_STAMP.to_le_bytes()
        {
            continue;
        }
        if let Some(road) = parse_to_road(data, offset) {
            roads.push(road);
        }
    }
    roads
}

fn select_road(tag: &Tag, roads: &[ToRoad]) -> Result<Vec<SelectedNavVector>, String> {
    let name = tag
        .attr("name")
        .ok_or_else(|| "course road is missing name".to_string())?;
    let start = tag.attr("start").unwrap_or("");
    let end = tag.attr("end").unwrap_or("");
    let road = roads
        .iter()
        .find(|road| road.name == name)
        .ok_or_else(|| format!("ToRoad not found: {name}"))?;
    let first = if start.is_empty() {
        0
    } else {
        road.elements
            .iter()
            .position(|element| element.marker == start)
            .ok_or_else(|| format!("{name}: start marker not found: {start}"))?
    };
    let last = if end.is_empty() {
        road.elements.len().saturating_sub(1)
    } else {
        road.elements[first..]
            .iter()
            .position(|element| element.marker == end)
            .map(|relative| first + relative)
            // Branch endpoints are commonly logical course markers rather than
            // duplicate per-road element labels. In that case the final element
            // is the arm's terminal vector and branch assembly owns the merge.
            .unwrap_or_else(|| road.elements.len().saturating_sub(1))
    };
    if road.elements.is_empty() || first > last {
        return Err(format!("{name}: empty or reversed element range"));
    }
    Ok((first..=last)
        .map(|element| SelectedNavVector {
            road: name.to_string(),
            element,
            records: road.elements[element].records.clone(),
        })
        .collect())
}

#[derive(Default)]
struct CourseFragment {
    contexts: Vec<SelectedNavVector>,
    successors: Vec<Vec<usize>>,
    entries: Vec<usize>,
    exits: Vec<usize>,
}

impl CourseFragment {
    fn from_contexts(contexts: Vec<SelectedNavVector>) -> Self {
        let count = contexts.len();
        let mut successors = vec![Vec::new(); count];
        for (index, next) in successors
            .iter_mut()
            .enumerate()
            .take(count.saturating_sub(1))
        {
            next.push(index + 1);
        }
        Self {
            contexts,
            successors,
            entries: (count > 0).then_some(0).into_iter().collect(),
            exits: (count > 0).then_some(count - 1).into_iter().collect(),
        }
    }

    fn append_parallel(&mut self, mut other: CourseFragment) -> (Vec<usize>, Vec<usize>) {
        let offset = self.contexts.len();
        for successors in &mut other.successors {
            for successor in successors {
                *successor += offset;
            }
        }
        let entries = other
            .entries
            .into_iter()
            .map(|index| index + offset)
            .collect::<Vec<_>>();
        let exits = other
            .exits
            .into_iter()
            .map(|index| index + offset)
            .collect::<Vec<_>>();
        self.contexts.extend(other.contexts);
        self.successors.extend(other.successors);
        (entries, exits)
    }

    fn append_sequence(&mut self, other: CourseFragment) {
        if other.contexts.is_empty() {
            return;
        }
        let prior_exits = self.exits.clone();
        let was_empty = self.contexts.is_empty();
        let (entries, exits) = self.append_parallel(other);
        if was_empty {
            self.entries = entries;
        } else {
            for exit in prior_exits {
                self.successors[exit].extend(entries.iter().copied());
            }
        }
        self.exits = exits;
    }

    fn detach_last_context(&mut self) -> Result<SelectedNavVector, String> {
        let terminal_index = self
            .contexts
            .len()
            .checked_sub(1)
            .ok_or_else(|| "course branch arm is empty".to_string())?;
        if self.successors[terminal_index].iter().any(|&next| next != terminal_index) {
            return Err("course branch terminal has an outgoing edge".to_string());
        }
        let mut predecessors = Vec::new();
        for (index, successors) in self.successors.iter_mut().enumerate() {
            if index == terminal_index {
                continue;
            }
            if successors.contains(&terminal_index) {
                successors.retain(|&next| next != terminal_index);
                predecessors.push(index);
            }
        }
        let terminal = self.contexts.pop().unwrap();
        self.successors.pop();
        self.entries.retain(|&index| index != terminal_index);
        self.exits = if self.contexts.is_empty() {
            Vec::new()
        } else if predecessors.is_empty() {
            self.exits
                .iter()
                .copied()
                .filter(|&index| index != terminal_index)
                .collect()
        } else {
            predecessors.clone()
        };
        Ok(terminal)
    }
}

fn assemble_tag_graph(tag: &Tag, roads: &[ToRoad]) -> Result<CourseFragment, String> {
    match tag.name.as_str() {
        "road" => Ok(CourseFragment::from_contexts(select_road(tag, roads)?)),
        "branch" => {
            let mut output = CourseFragment::default();
            let mut branch_entries = Vec::new();
            let mut branch_exits = Vec::new();
            let mut canonical_terminal = None;
            let mut has_terminal_only_arm = false;
            for arm in &tag.children {
                let mut selected = assemble_tag_graph(arm, roads)?;
                let terminal = selected.detach_last_context()?;
                let arm_was_terminal_only = selected.contexts.is_empty();
                let (entries, exits) = output.append_parallel(selected);
                if arm_was_terminal_only {
                    has_terminal_only_arm = true;
                } else {
                    branch_entries.extend(entries);
                    branch_exits.extend(exits);
                }
                if canonical_terminal.is_none() {
                    canonical_terminal = Some(terminal);
                }
            }
            if let Some(terminal) = canonical_terminal {
                let terminal_index = output.contexts.len();
                output.contexts.push(terminal);
                output.successors.push(Vec::new());
                for exit in &branch_exits {
                    output.successors[*exit].push(terminal_index);
                }
                if has_terminal_only_arm || branch_entries.is_empty() {
                    branch_entries.push(terminal_index);
                }
                output.entries = branch_entries;
                output.exits = vec![terminal_index];
            }
            Ok(output)
        }
        _ => {
            let mut output = CourseFragment::default();
            for child in &tag.children {
                if matches!(child.name.as_str(), "road" | "branch" | "course") {
                    output.append_sequence(assemble_tag_graph(child, roads)?);
                }
            }
            Ok(output)
        }
    }
}

/// Assemble the registered forward/reverse navigation-vector order from an
/// embedded `course` tag and decoded `ToRoad` objects.
///
/// Parallel branch arms publish their non-terminal vectors in author order,
/// then share the first arm's terminal vector. This reproduces the legacy
/// `sub_A50500` branch merge contract without consulting process memory.
pub fn assemble_course(course: &Tag, roads: &[ToRoad]) -> Result<Vec<SelectedNavVector>, String> {
    Ok(assemble_course_graph(course, roads)?.contexts)
}

/// Assemble navigation contexts together with the authored successor graph
/// consumed by the runtime context `+0x08` successor vector. Parallel branch
/// arms share their canonical terminal, and the root course is a closed lap.
pub fn assemble_course_graph(course: &Tag, roads: &[ToRoad]) -> Result<AssembledCourse, String> {
    (course.name == "course" || course.name == "courseReverse")
        .then_some(())
        .ok_or_else(|| format!("expected course tag, got {}", course.name))?;
    let mut graph = assemble_tag_graph(course, roads)?;
    for exit in graph.exits.clone() {
        graph.successors[exit].extend(graph.entries.iter().copied());
    }
    Ok(AssembledCourse {
        contexts: graph.contexts,
        successors: graph.successors,
    })
}

/// Discover complete embedded 36-byte navigation-record vectors.
///
/// A valid KartRider road path contains at least two records.  The upper bound
/// protects malformed assets from asking for an unbounded allocation; observed
/// retail tracks are far below it.
pub fn discover_record_vectors(data: &[u8]) -> Vec<NavRecordVector> {
    const MIN_RECORDS: usize = 2;
    const MAX_RECORDS: usize = 4096;
    let mut vectors = Vec::new();
    for offset in 0..data.len().saturating_sub(4) {
        let count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        if !(MIN_RECORDS..=MAX_RECORDS).contains(&count) {
            continue;
        }
        let Some(byte_len) = count.checked_mul(RECORD_SIZE) else {
            continue;
        };
        let Some(end) = offset
            .checked_add(4)
            .and_then(|start| start.checked_add(byte_len))
        else {
            continue;
        };
        if end > data.len() {
            continue;
        }
        let records = (0..count)
            .map(|index| decode_record(data, offset + 4 + index * RECORD_SIZE))
            .collect::<Vec<_>>();
        if records.iter().all(physically_valid) {
            vectors.push(NavRecordVector { offset, records });
        }
    }
    vectors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_record(data: &mut Vec<u8>, position: Vec3, secondary: Vec3, side: Vec3) {
        for value in position.into_iter().chain(secondary).chain(side) {
            data.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn push_text(data: &mut Vec<u8>, text: &str) {
        let units = text.encode_utf16().collect::<Vec<_>>();
        data.extend_from_slice(&(units.len() as u32).to_le_bytes());
        for unit in units {
            data.extend_from_slice(&unit.to_le_bytes());
        }
    }

    #[test]
    fn discovers_unaligned_record_vector() {
        let mut data = vec![0xAA];
        data.extend_from_slice(&2u32.to_le_bytes());
        push_record(&mut data, [1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        push_record(&mut data, [4.0, 5.0, 6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let vectors = discover_record_vectors(&data);
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].offset, 1);
        assert_eq!(vectors[0].records[1].position, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_non_unit_or_non_orthogonal_bases() {
        let mut data = 2u32.to_le_bytes().to_vec();
        for _ in 0..2 {
            push_record(&mut data, [1.0, 2.0, 3.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        }
        assert!(discover_record_vectors(&data).is_empty());
    }

    #[test]
    fn decodes_to_road_ownership_and_element_fields() {
        let mut data = Vec::new();
        data.extend_from_slice(&0x47AAu16.to_le_bytes());
        data.extend_from_slice(&TO_ROAD_STAMP.to_le_bytes());
        data.extend_from_slice(&7u16.to_le_bytes());
        push_text(&mut data, "RoadObj01");
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        push_text(&mut data, "start");
        data.extend_from_slice(&3u32.to_le_bytes());
        for point in [[0.0f32, 0.0, 0.0], [1.0f32, 0.0, 0.0], [0.0f32, 1.0, 0.0]] {
            for value in point {
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        data.extend_from_slice(&1u32.to_le_bytes());
        for index in [0u16, 1, 2] {
            data.extend_from_slice(&index.to_le_bytes());
        }
        push_text(&mut data, "rail");
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        for x in [0.0, 1.0] {
            push_record(&mut data, [x, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        }

        let roads = discover_to_roads(&data);
        assert_eq!(roads.len(), 1);
        assert_eq!(roads[0].name, "RoadObj01");
        assert_eq!(roads[0].elements[0].marker, "start");
        assert_eq!(roads[0].elements[0].surface, "rail");
        assert_eq!(roads[0].elements[0].triangles, vec![[0, 1, 2]]);
        assert_eq!(roads[0].elements[0].records.len(), 2);
    }

    fn nav_element(marker: &str, x: f32) -> ToRoadElement {
        ToRoadElement {
            marker: marker.into(),
            boundary: Vec::new(),
            triangles: Vec::new(),
            surface: String::new(),
            surface_triangles: Vec::new(),
            records: vec![NavRecord {
                position: [x, 0.0, 0.0],
                secondary: [1.0, 0.0, 0.0],
                normal_or_side: [0.0, 0.0, 1.0],
            }],
        }
    }

    fn road_tag(name: &str, start: &str, end: &str) -> Tag {
        Tag {
            name: "road".into(),
            attributes: vec![
                ("name".into(), name.into()),
                ("start".into(), start.into()),
                ("end".into(), end.into()),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn assembles_parallel_arms_with_first_terminal_as_merge_owner() {
        let roads = vec![
            ToRoad {
                offset: 0,
                name: "main".into(),
                flags: 0,
                elements: vec![
                    nav_element("start", 0.0),
                    nav_element("", 1.0),
                    nav_element("split", 2.0),
                    nav_element("join", 9.0),
                ],
            },
            ToRoad {
                offset: 0,
                name: "left".into(),
                flags: 0,
                elements: vec![nav_element("start", 3.0), nav_element("", 4.0)],
            },
            ToRoad {
                offset: 0,
                name: "right".into(),
                flags: 0,
                elements: vec![nav_element("start", 5.0), nav_element("", 6.0)],
            },
        ];
        let course = Tag {
            name: "course".into(),
            children: vec![
                road_tag("main", "start", "split"),
                Tag {
                    name: "branch".into(),
                    children: vec![
                        Tag {
                            name: "course".into(),
                            children: vec![road_tag("left", "start", "end")],
                            ..Default::default()
                        },
                        Tag {
                            name: "course".into(),
                            children: vec![road_tag("right", "start", "end")],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                road_tag("main", "join", "join"),
            ],
            ..Default::default()
        };
        let selected = assemble_course(&course, &roads).unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|vector| vector.records[0].position[0])
                .collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 5.0, 4.0, 9.0]
        );
        let graph = assemble_course_graph(&course, &roads).unwrap();
        assert_eq!(graph.successors[2], vec![3, 4]);
        assert_eq!(graph.successors[3], vec![5]);
        assert_eq!(graph.successors[4], vec![5]);
        assert_eq!(graph.successors[5], vec![6]);
        assert_eq!(graph.successors[6], vec![0]);
    }
}
