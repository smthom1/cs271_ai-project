# code to run tests, all from https://minesweeper.online/help/patterns
import numpy as np
from constants import CLOSED, FLAG
from agent_eval import solve_csp


# run a bunch of trivial test cases to see if the agent logic is sound

def print_board(board):
    print("\nBoard:")
    for row in board:
        print(" ".join(f"{x:2}" for x in row))
    print()

# put test funcs into test wrapper

TEST_RESULTS = []
flagTests = 0
safeTests = 0
flagsPassed = 0
safePassed = 0

def run_test(name, testfunc, testType):
    print("--------------------------------")
    global flagTests
    global safeTests
    global flagsPassed
    global safePassed
    if testType == "flag":
        flagTests += 1

    if testType == "safe":
        safeTests += 1

    try:
        testfunc()
        TEST_RESULTS.append((name, True))
        if testType == "flag":
            flagsPassed += 1

        if testType == "safe":
            safePassed += 1
        

    except AssertionError as e:
        print(f"❌ FAILED: {e}")

        
        TEST_RESULTS.append((name, False))

    except Exception as e:
        print(f"❌ ERROR during {name}: {e}")
        TEST_RESULTS.append((name, False))

    
    print("--------------------------------")

# all the tests

def test_single_safe_move():
    board = np.array([
        [CLOSED, 1, CLOSED],
        [1, FLAG, 1],
        [CLOSED, 1, CLOSED]
    ])

    print("Test: mine surrounded with ones, find the safe spot")
    print_board(board)

    safe = solve_csp(board)[0]
    expected = {(2, 2), (0, 2), (2, 0), (0, 0)}

    print("Safe Moves Found:", safe)
    assert expected == safe, ("Wrong, expected result is ", expected)
    print("PASSED TEST\n")


def test_single_flag_move():
    board = np.array([
        [1, 1, 1],
        [1, CLOSED, 1],
        [1, 1, 1]
    ])

    print("\nTEST: Single Flag Move")
    print_board(board)

    flag_moves = solve_csp(board)[1]
    expected = {(1, 1)}

    print("Flag Moves Found:", flag_moves)
    assert expected == flag_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")


def onetwoonetest():
    board = np.array([
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [1, 1, 2, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    print("\nTEST: 1-2-1 test")
    print_board(board)

    flag_moves = solve_csp(board)[1]
    expected = {(1, 1), (1, 3)}

    print("Flag Moves Found:", flag_moves)
    assert expected == flag_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")


def oneonetest():
    board = np.array([
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    print("\nTEST: 1-1 Test")
    print_board(board)

    safe_moves = solve_csp(board)[0]
    expected = {(1, 2)}

    print("Safe Moves Found:", safe_moves)
    assert expected == safe_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")

# error if more rows than columns
def oneoneplustest():
    board = np.array([
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [CLOSED, 2, CLOSED, CLOSED, CLOSED],
        [1, 1, 1, 1, CLOSED],
        [0, 0, 0, 1, CLOSED],
        [0, 0, 0, 1, CLOSED]
    ])

    print("\nTEST: 1-1+ Test")
    print_board(board)

    safe_moves = solve_csp(board)[0]
    expected = {(1, 4), (2, 4), (3, 4)}

    print("Safe Moves Found:", safe_moves)
    assert expected == safe_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")


def b1test():
    board = np.array([
        [0, 0, CLOSED, CLOSED, CLOSED],
        [0, 0, 2, CLOSED, CLOSED],
        [0, 0, 3, CLOSED, CLOSED],
        [0, 0, 2, CLOSED, CLOSED],
        [0, 0, CLOSED, CLOSED, CLOSED]
    ])

    print("\nTEST: b1 Test")
    print_board(board)

    flag_moves = solve_csp(board)[1]
    expected = {(1, 3), (2, 3), (3, 3)}

    print("Flag Moves Found:", flag_moves)
    assert expected == flag_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")


def b2test():
    board = np.array([
        [0, 0, 2, -3, CLOSED],
        [0, 0, 3, -3, CLOSED],
        [1, 1, 2, -3, CLOSED],
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED]
    ])

    print("\nTEST: b2 Test")
    print_board(board)

    safe_moves = solve_csp(board)[0]
    expected = {(3, 1), (3, 2), (3, 3)}

    print("Safe Moves Found:", safe_moves)
    assert expected == safe_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")

# weird issue: if there is fifith row of 0s then the ai can't find anything.
def onetwotest():
    board = np.array([
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [1, 2, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ])

    print("\nTEST: 1-2 Test")
    print_board(board)

    flag_moves = solve_csp(board)[1]
    expected = {(1,2), (1,0)}

    print("Flag Moves Found:", flag_moves)
    assert expected == flag_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")


def no_good_moves():
    board = np.array([
        [1, CLOSED, CLOSED],
        [CLOSED, CLOSED, CLOSED],
        [CLOSED, CLOSED, CLOSED]
    ])

    print("\nTEST: no good moves")
    print_board(board)

    safe = solve_csp(board)[0]
    flag = solve_csp(board)[1]

    print("Safe Moves:", safe)
    print("Flag Moves:", flag)

    expected_safe = set()
    expected_flag = set()

    assert len(safe) == 0 and len(flag) == 0, ("Should be no good moves found", (expected_safe, expected_flag))
    print("PASSED TEST\n")

# should we pass if flag more than what's needed?
def onetwoplustest():
    board = np.array([
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [CLOSED, 2, CLOSED, CLOSED, CLOSED],
        [1, 1, 1, 4, CLOSED],
        [0, 0, 0, 2, CLOSED],
        [0, 0, 0, 1, CLOSED]
    ])

    print("\nTEST: 1-2+ Test")
    print_board(board)

    flag_moves = solve_csp(board)[1]
    expected = {(1,4), (2,4), (3,4), (1,0)}

    print("Flag Moves Found:", flag_moves)
    assert expected == (flag_moves), ("Wrong, expected result is", expected)
    print("PASSED TEST\n")

def onetwoCtest():
    board = np.array([
        
        [ CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [ 3, 1, 2, 3, CLOSED],
        [ 0, 0, 0, 2, CLOSED],
        [ 0, 0, 0, 3, CLOSED],
        [ CLOSED, CLOSED, CLOSED, CLOSED, CLOSED]
    ])

    print("\nTEST: 1-2C Test")
    print_board(board)

    flag_moves = solve_csp(board)[1]
    expected = {(0,3)}

    print("Flag Moves Found:", flag_moves)
    assert expected == (flag_moves), ("Wrong, expected result is", expected)
    print("PASSED TEST\n")

def onetwotwoonetest():
    board = np.array([
        
        [  CLOSED, CLOSED, CLOSED, CLOSED],
        [  CLOSED, CLOSED, CLOSED, CLOSED],
        [  1, 2, 2, 1],
        [  0, 0, 0, 0]
    ])

    print("\nTEST: 1-2-2-1 Test")
    print_board(board)

    flag_moves = solve_csp(board)[0]
    expected = {(1,1), (1,2)}

    print("Flag Moves Found:", flag_moves)
    assert expected == (flag_moves), ("Wrong, expected result is", expected)
    print("PASSED TEST\n")

def oneoneR():
    board = np.array([
        [0, 1, CLOSED, CLOSED, CLOSED],
        [1, 2, CLOSED, CLOSED, CLOSED],
        [-3, 2, CLOSED, CLOSED, CLOSED],
        [3, 4, CLOSED, CLOSED, CLOSED],
        [-3, -3, CLOSED, CLOSED, CLOSED]
    ])

    print("\nTEST: 1-1R Test")
    print_board(board)

    safe_moves = solve_csp(board)[0]
    expected = {(1,2)}

    print("Safe Moves Found:", safe_moves)
    assert expected == safe_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")

def onetwoR():
    board = np.array([
        [3, 2, 1, 1, 2],
        [-3, -3, 1, 1, -3],
        [CLOSED, 4, 2, 3, 3],
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED]
    ])

    print("\nTEST: 1-2R Test")
    print_board(board)

    flag_moves = solve_csp(board)[1]
    expected = {(3,4)}

    print("Flag Moves Found:", flag_moves)
    assert expected == flag_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")

def onetwooneR():
    board = np.array([
         [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
         [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [1, 2, 3, 2, 1],
        [0, 1, -3, 1, 0],
        [0, 1, 1, 1, 0]
    ])

    print("\nTEST: 1-2-1R Test")
    print_board(board)

    flag_moves = solve_csp(board)[1]
    expected = {(1,1),(1,3)}

    print("Flag Moves Found:", flag_moves)
    assert expected == flag_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")

def h1():
    board = np.array([
         [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
         [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
        [CLOSED, CLOSED, 1, CLOSED, CLOSED],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ])

    print("\nTEST: H1 Test")
    print_board(board)

    safe_moves = solve_csp(board)[0]
    expected = {(1,1),(1,2),(1,3)}

    print("Safe Moves Found:",safe_moves)
    assert expected == safe_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")

def h2():
    board = np.array([
         [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
         [CLOSED, 2, 1, 3, CLOSED],
        [CLOSED, CLOSED, 1, CLOSED, CLOSED],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ])

    print("\nTEST: H2 Test")
    print_board(board)

    safe_moves = solve_csp(board)[0]
    expected = {(0,1),(0,2),(0,3)}

    print("Safe Moves Found:",safe_moves)
    assert expected == safe_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")

# weird thing, changing 3 to one gives a result set 
def h3():
    board = np.array([
         [CLOSED, CLOSED, CLOSED, CLOSED, CLOSED],
         [CLOSED, 2, 3, 1, CLOSED],
        [CLOSED, CLOSED, 1, CLOSED, CLOSED],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ])

    print("\nTEST: H3 Test")
    print_board(board)

    safe_moves = solve_csp(board)[0]
    expected = {(0,2),(0,3),(0,4),(1,4)}

    print("Safe Moves Found:",safe_moves)
    assert expected == safe_moves, ("Wrong, expected result is", expected)
    print("PASSED TEST\n")

if __name__ == "__main__":
    # global flagTests
    # global safeTests
    # global flagsPassed
    # global safePassed
    print("Testing testing 123\n")

    #easy stuff
    run_test("Single Safe Move", test_single_safe_move, "safe")
    run_test("Single Flag Move", test_single_flag_move, "flag")

    run_test("No Good Moves", no_good_moves, None)

    # stuff from mineswepper online
    run_test("b1 test", b1test, "flag")
    run_test("b2 test", b2test, "safe")
    
    run_test("1-1 Test", oneonetest, "safe")
    run_test("1-1+ Test", oneoneplustest, "safe")
    run_test("1-2 test", onetwotest, "flag")

    run_test("1-2+ Test", onetwoplustest, "flag")
    run_test("1-2C Test", onetwoCtest, "flag")
    run_test("1-2-1 Test", onetwoonetest, "flag")

    run_test("1-2-2-1 Test", onetwotwoonetest, "flag")
    run_test("1-1R Test", oneoneR, "safe")
    run_test("1-2R Test", onetwoR, "flag")
    run_test("1-2-1R Test", onetwooneR, "flag")

    run_test("H1 Test", h1, "safe")
    run_test("H2 Test", h2, "safe")
    run_test("H3 Test", h3, "safe")
    

    

    passed = sum(result[1] for result in TEST_RESULTS)
    pct = (passed / len(TEST_RESULTS)) * 100

    print(f"TEST SUMMARY: {passed}/{len(TEST_RESULTS)} PASSED ({pct:.1f}%)")
    print(f"We passed {(safePassed/safeTests * 100):.1f}% tests to find safe spots")
    print(f"We passed {(flagsPassed/flagTests * 100):.1f}% tests to flag mines")

