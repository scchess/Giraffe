TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    giraffe.cpp \
    magic_moves.cpp \
    board_consts.cpp \
    board.cpp

HEADERS += \
    board_consts.h \
    magic_moves.h \
    types.h \
    board.h \
    move.h \
    bit_ops.h \
    containers.h \
    util.h
