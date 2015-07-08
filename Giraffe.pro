TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += $$PWD/eval
INCLUDEPATH += $$PWD/ann
INCLUDEPATH += $$PWD

SOURCES += main.cpp \
	giraffe.cpp \
	magic_moves.cpp \
	board_consts.cpp \
	board.cpp \
	eval/eval.cpp \
	search.cpp \
	see.cpp \
	backend.cpp \
	chessclock.cpp \
	timeallocator.cpp \
	zobrist.cpp \
	ttable.cpp \
	killer.cpp \
	movepicker.cpp \
	ann/ann.cpp \
	tools/gen_bitboard_consts.cpp \
	ann/learn_ann.cpp \
	ann/features_conv.cpp \
    learn.cpp \
    random_device.cpp

HEADERS += \
	board_consts.h \
	magic_moves.h \
	types.h \
	board.h \
	move.h \
	bit_ops.h \
	containers.h \
	util.h \
	eval/eval.h \
	search.h \
	see.h \
	backend.h \
	chessclock.h \
	timeallocator.h \
	eval/eval_params.h \
	zobrist.h \
	ttable.h \
	killer.h \
	movepicker.h \
	ann/ann.h \
	ann/ann_impl.h \
	ann/learn_ann.h \
	ann/features_conv.h \
	matrix_ops.h \
	evaluator.h \
	ann/ann_evaluator.h \
    learn.h \
    omp_scoped_thread_limiter.h \
    random_device.h

