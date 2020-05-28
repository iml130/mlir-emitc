configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(EMITC_TEST_DEPENDS
        FileCheck count not
        emitc-opt
        emitc-translate
        )

add_lit_testsuite(check-emitc "Running the EmitC regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${EMITC_TEST_DEPENDS}
        )
set_target_properties(check-emitc PROPERTIES FOLDER "Tests")

add_lit_testsuites(STANDALONE_TEST ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${EMITC_TEST_DEPENDS})
