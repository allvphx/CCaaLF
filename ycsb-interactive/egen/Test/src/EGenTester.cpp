/*
 * Legal Notice
 *
 * This document and associated source code (the "Work") is a part of a
 * benchmark specification maintained by the TPC.
 *
 * The TPC reserves all right, title, and interest to the Work as provided
 * under U.S. and international laws, including without limitation all patent
 * and trademark rights therein.
 *
 * No Warranty
 *
 * 1.1 TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE INFORMATION
 *     CONTAINED HEREIN IS PROVIDED "AS IS" AND WITH ALL FAULTS, AND THE
 *     AUTHORS AND DEVELOPERS OF THE WORK HEREBY DISCLAIM ALL OTHER
 *     WARRANTIES AND CONDITIONS, EITHER EXPRESS, IMPLIED OR STATUTORY,
 *     INCLUDING, BUT NOT LIMITED TO, ANY (IF ANY) IMPLIED WARRANTIES,
 *     DUTIES OR CONDITIONS OF MERCHANTABILITY, OF FITNESS FOR A PARTICULAR
 *     PURPOSE, OF ACCURACY OR COMPLETENESS OF RESPONSES, OF RESULTS, OF
 *     WORKMANLIKE EFFORT, OF LACK OF VIRUSES, AND OF LACK OF NEGLIGENCE.
 *     ALSO, THERE IS NO WARRANTY OR CONDITION OF TITLE, QUIET ENJOYMENT,
 *     QUIET POSSESSION, CORRESPONDENCE TO DESCRIPTION OR NON-INFRINGEMENT
 *     WITH REGARD TO THE WORK.
 * 1.2 IN NO EVENT WILL ANY AUTHOR OR DEVELOPER OF THE WORK BE LIABLE TO
 *     ANY OTHER PARTY FOR ANY DAMAGES, INCLUDING BUT NOT LIMITED TO THE
 *     COST OF PROCURING SUBSTITUTE GOODS OR SERVICES, LOST PROFITS, LOSS
 *     OF USE, LOSS OF DATA, OR ANY INCIDENTAL, CONSEQUENTIAL, DIRECT,
 *     INDIRECT, OR SPECIAL DAMAGES WHETHER UNDER CONTRACT, TORT, WARRANTY,
 *     OR OTHERWISE, ARISING IN ANY WAY OUT OF THIS OR ANY OTHER AGREEMENT
 *     RELATING TO THE WORK, WHETHER OR NOT SUCH AUTHOR OR DEVELOPER HAD
 *     ADVANCE NOTICE OF THE POSSIBILITY OF SUCH DAMAGES.
 *
 * Contributors
 * - Doug Johnson
 */

//
// In order to use the dynamic library, define BOOST_TEST_DYN_LINK
// prior to including unit_test.hpp. This needs to be defined for each
// module in the application that utilizes BOOST. So rather than define
// it here, it is defined at the project level (i.e. in the makefile or 
// Visual Studio project).
// #define BOOST_TEST_DYN_LINK
//

#include <boost/test/unit_test.hpp>

#include "../inc/TestSuiteBuilder.h"
#include "../../Utilities/Test/inc/EGenUtilitiesTestSuite.h"
#include "../../InputFiles/Test/inc/EGenInputFilesTestSuite.h"

using namespace EGenTestCommon;
using namespace EGenUtilitiesTest;
using namespace EGenInputFilesTest;

bool
init_function()
{
    //
    // Set the logging level. This controls the level of detail in BOOST output.
    //
    boost::unit_test::unit_test_log.set_threshold_level( boost::unit_test::log_test_units );

    //
    // Set the name for the master test suite.
    //
    boost::unit_test::framework::master_test_suite().p_name.set( "EGen Master Test Suite" );

    //
    // Load EGenUtilities Test Suite
    //
    TestSuiteBuilder< EGenUtilitiesTestSuite > utilitiesTestSuite( "EGenUtilities Test Suite" );
    boost::unit_test::framework::master_test_suite().add( utilitiesTestSuite.TestSuite() );

    //
    // Load EGenInputFiles Test Suite
    //
    TestSuiteBuilder< EGenInputFilesTestSuite > inputFilesTestSuite( "EGenInputFiles Test Suite" );
    boost::unit_test::framework::master_test_suite().add( inputFilesTestSuite.TestSuite() );

    return true;
}

//___________________________________________________________________________//

int
main( int argc, char* argv[] )
{
    return ::boost::unit_test::unit_test_main( &init_function, argc, argv );
}
