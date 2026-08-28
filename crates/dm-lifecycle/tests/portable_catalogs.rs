use std::collections::BTreeMap;

use dm_lifecycle::{
    decode_dmm_measurements, decode_parsed_dmm_cache, encode_dmm_measurements,
    encode_parsed_dmm_cache, PortableDmmGrid, PortableDmmMeasurement, PortableParsedDmm,
};
use dm_vm::DmmMeasurement;

#[test]
fn dmm_measurement_catalog_round_trips() {
    let digest = [0x42; 16];
    let measurement = PortableDmmMeasurement {
        digest,
        measurement: DmmMeasurement {
            digest,
            bounds: [1, 2, 3, 10, 20, 4],
        },
    };
    let mut catalog = BTreeMap::new();
    catalog.insert("maps/test.dmm".to_owned(), measurement.clone());

    let encoded = encode_dmm_measurements(&catalog).expect("encode catalog");
    let decoded = decode_dmm_measurements(&encoded).expect("decode catalog");

    assert_eq!(decoded.get("maps/test.dmm"), Some(&measurement));
}

#[test]
fn parsed_dmm_catalog_round_trips() {
    let digest = [0x24; 16];
    let entry = PortableParsedDmm {
        digest,
        tgm: true,
        key_len: 2,
        line_len: 4,
        bounds: [1, 1, 1, 2, 2, 1],
        models: vec![("aa".to_owned(), "/turf".to_owned())],
        grids: vec![PortableDmmGrid {
            x: 1,
            y: 2,
            z: 1,
            lines: vec!["aabb".to_owned()],
        }],
    };
    let mut catalog = BTreeMap::new();
    catalog.insert("maps/test.dmm".to_owned(), entry.clone());

    let encoded = encode_parsed_dmm_cache(&catalog).expect("encode parsed catalog");
    let decoded = decode_parsed_dmm_cache(&encoded).expect("decode parsed catalog");

    assert_eq!(decoded.get("maps/test.dmm"), Some(&entry));
}

#[test]
fn catalog_checksums_reject_tampering() {
    let digest = [0x11; 16];
    let measurement = PortableDmmMeasurement {
        digest,
        measurement: DmmMeasurement {
            digest,
            bounds: [1, 1, 1, 1, 1, 1],
        },
    };
    let mut catalog = BTreeMap::new();
    catalog.insert("test.dmm".to_owned(), measurement);

    let mut encoded = encode_dmm_measurements(&catalog).expect("encode catalog");
    *encoded.last_mut().expect("payload byte") ^= 0xff;

    let error = decode_dmm_measurements(&encoded).expect_err("tampered catalog must fail");
    assert!(error.contains("checksum"));
}
