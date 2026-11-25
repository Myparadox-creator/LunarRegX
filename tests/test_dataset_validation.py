from pathlib import Path
import tempfile
import pytest
import numpy as np
from PIL import Image

from src.io.dataset import load_lunar_image, parse_lunar_xml, validate_lunar_xml

SAMPLE_PDS4_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1" xmlns:isda="https://isda.issdc.gov.in/pds4/isda/v1">
    <Identification_Area>
        <logical_identifier>urn:isro:isda:ch2_cho.ohr:data_calibrated:ch2_ohr_ncp_20190906t2241285714_d_img_gds</logical_identifier>
        <title>Chandrayaan-2 Orbiter OHRC Experiment</title>
    </Identification_Area>
    <Observation_Area>
        <Mission_Area>
            <isda:Product_Parameters>
                <isda:job_id>OHRXXGDSCHO0015702NNNN19249235207985_V1_0</isda:job_id>
                <isda:pixel_resolution unit="m/pixel">0.25</isda:pixel_resolution>
                <isda:sun_azimuth unit="deg">67.0</isda:sun_azimuth>
                <isda:sun_elevation unit="deg">6.3</isda:sun_elevation>
                <isda:solar_incidence unit="deg">83.7</isda:solar_incidence>
            </isda:Product_Parameters>
            <isda:Geometry_Parameters>
                <isda:Refined_Corner_Coordinates>
                    <isda:upper_left_latitude>-70.55</isda:upper_left_latitude>
                    <isda:upper_left_longitude>22.57</isda:upper_left_longitude>
                    <isda:lower_right_latitude>-71.39</isda:lower_right_latitude>
                    <isda:lower_right_longitude>22.92</isda:lower_right_longitude>
                </isda:Refined_Corner_Coordinates>
            </isda:Geometry_Parameters>
        </Mission_Area>
    </Observation_Area>
    <File_Area_Observational>
        <File><file_name>ch2_ohr_ncp_20190906T2241285714_d_img_gds.img</file_name></File>
    </File_Area_Observational>
</Product_Observational>
"""

SAMPLE_SAC_XML = """<?xml version="1.0" encoding="utf-8"?>
<product>
    <job_id>IIRXXD32CHO1519402NNNN23022110758606_V1_1_01</job_id>
    <level0_dataset>IIRXXD32CHO1519402NNNN23022110758606_V1_1_01</level0_dataset>
    <Resolution_in_meter>83.14</Resolution_in_meter>
    <Sun_azimuth_in_degree>45.45</Sun_azimuth_in_degree>
    <Sun_elevation_in_degree>21.05</Sun_elevation_in_degree>
    <Solar_incidence_angle_in_degree>68.94</Solar_incidence_angle_in_degree>
    <iir>
        <corners>
            <topleft_latitude>-70.0</topleft_latitude>
            <topleft_longitude>31.0</topleft_longitude>
            <bottomright_latitude>-71.0</bottomright_latitude>
            <bottomright_longitude>33.0</bottomright_longitude>
        </corners>
    </iir>
</product>
"""

def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_lunar_image('non_existent_file_path.tif')

def test_empty_file_raises_error(tmp_path):
    empty_file = tmp_path / 'empty.tif'
    empty_file.write_bytes(b'')
    with pytest.raises(ValueError, match='0 bytes'):
        load_lunar_image(empty_file)

def test_corrupt_file_raises_error(tmp_path):
    corrupt_file = tmp_path / 'corrupt.png'
    corrupt_file.write_bytes(b'Not an image file content header')
    with pytest.raises(ValueError, match='Failed to decode'):
        load_lunar_image(corrupt_file)

def test_tiny_image_raises_error(tmp_path):
    tiny_file = tmp_path / 'tiny.png'
    tiny_arr = np.zeros((4, 4), dtype=np.uint8)
    Image.fromarray(tiny_arr).save(tiny_file)
    with pytest.raises(ValueError, match='too small'):
        load_lunar_image(tiny_file)

def test_valid_image_loads_successfully(tmp_path):
    valid_file = tmp_path / 'valid.png'
    arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
    Image.fromarray(arr).save(valid_file)
    lunar_img = load_lunar_image(valid_file, gsd_override=0.25)
    assert lunar_img.width == 64
    assert lunar_img.height == 64
    assert lunar_img.metadata.gsd == 0.25
    assert lunar_img.display_8bit.shape == (64, 64)

def test_require_xml_blocks_missing_xml(tmp_path):
    img_path = tmp_path / 'custom_src.png'
    arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
    Image.fromarray(arr).save(img_path)
    with pytest.raises(ValueError, match='Missing corresponding XML'):
        load_lunar_image(img_path, require_xml=True)

def test_parse_and_validate_pds4_and_sac_xml(tmp_path):
    pds4_xml = tmp_path / 'ch2_ohr_ncp_20190906T2241285714_d_img_gds.xml'
    pds4_xml.write_text(SAMPLE_PDS4_XML, encoding='utf-8')
    sac_xml = tmp_path / 'IIRXXD32CHO1519402NNNN23022110758606_V1_1_01_reference.xml'
    sac_xml.write_text(SAMPLE_SAC_XML, encoding='utf-8')

    # PDS4 parse + matching browse PNG name with different stem suffix
    m_pds4 = validate_lunar_xml(pds4_xml, image_filename='ch2_ohr_ncp_20190906T2241285714_b_brw_gds.png')
    assert m_pds4['xml_loaded'] is True
    assert m_pds4['xml_format'] == 'PDS4'
    assert m_pds4['sensor_name'] == 'OHRC'
    assert m_pds4['solar_azimuth_deg'] == pytest.approx(67.0)

    # SAC parse + matching TIFF name
    m_sac = validate_lunar_xml(sac_xml, image_filename='IIRXXD32CHO1519402NNNN23022110758606_V1_1_01_reference.tif')
    assert m_sac['xml_loaded'] is True
    assert m_sac['xml_format'] == 'SAC_PRODUCT'
    assert m_sac['sensor_name'] == 'IIRS'
    assert m_sac['gsd'] == pytest.approx(83.14)
    assert m_sac['solar_azimuth_deg'] == pytest.approx(45.45)

    # Custom/generic filename accepts valid lunar XML
    m_custom = validate_lunar_xml(pds4_xml, image_filename='custom_lunar_crop.png')
    assert m_custom['xml_loaded'] is True

def test_validate_xml_rejects_invalid_and_mismatched_xml(tmp_path):
    bad_syntax = tmp_path / 'broken.xml'
    bad_syntax.write_text('<Product_Observational><unclosed>', encoding='utf-8')
    with pytest.raises(ValueError, match='Malformed XML'):
        validate_lunar_xml(bad_syntax)

    non_lunar = tmp_path / 'pom.xml'
    non_lunar.write_text('<project><modelVersion>4.0.0</modelVersion></project>', encoding='utf-8')
    with pytest.raises(ValueError, match='not a valid PDS4 or ISRO SAC'):
        validate_lunar_xml(non_lunar)

    pds4_xml = tmp_path / 'ch2_ohr_ncp_20190906T2241285714_d_img_gds.xml'
    pds4_xml.write_text(SAMPLE_PDS4_XML, encoding='utf-8')

    # Sensor mismatch: IIRS image paired with OHRC XML
    with pytest.raises(ValueError, match='XML correspondence mismatch'):
        validate_lunar_xml(pds4_xml, image_filename='IIRXXD32CHO1519402NNNN23022110758606_V1_1_01_reference.tif')

    # Observation timestamp mismatch: different OHRC pass
    with pytest.raises(ValueError, match='XML correspondence mismatch'):
        validate_lunar_xml(pds4_xml, image_filename='ch2_ohr_ncp_20211023T0027462822_b_brw_d18.png')

def test_load_lunar_image_with_explicit_xml(tmp_path):
    img_path = tmp_path / 'ch2_ohr_ncp_20190906T2241285714_b_brw_gds.png'
    xml_path = tmp_path / 'ch2_ohr_ncp_20190906T2241285714_d_img_gds.xml'
    arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
    Image.fromarray(arr).save(img_path)
    xml_path.write_text(SAMPLE_PDS4_XML, encoding='utf-8')

    loaded = load_lunar_image(img_path, xml_path=xml_path, require_xml=True)
    assert loaded.metadata.extra_attributes.get('xml_loaded') is True
    assert loaded.metadata.solar_azimuth_deg == pytest.approx(67.0)
