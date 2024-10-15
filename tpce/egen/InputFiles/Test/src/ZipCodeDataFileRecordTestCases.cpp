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
// Include this module's header first to make sure it is self-contained
//
#include "../inc/ZipCodeDataFileRecordTestCases.h"

//
// Include system headers
//
#include <stdlib.h>

//
// Include application headers
//
#include "../../../Test/inc/TestUtilities.h"
#include "../inc/Utilities.h"

using namespace EGenTestCommon;

namespace EGenInputFilesTest
{
    //
    // Constructor / Destructor
    //
    ZipCodeDataFileRecordTestCases::ZipCodeDataFileRecordTestCases()
        : dfr1( 0 )
        , divisionTaxKey("43")
        , zc_code("10001")
        , zc_town("New York")
        , zc_div("NY")
    {
        fields.push_back(divisionTaxKey);
        fields.push_back(zc_code);
        fields.push_back(zc_town);
        fields.push_back(zc_div);
    }

    ZipCodeDataFileRecordTestCases::~ZipCodeDataFileRecordTestCases()
    {
        CleanUp( &dfr1 );
    }

    //
    // Add test cases to the test suite.
    //
    void ZipCodeDataFileRecordTestCases::AddTestCases( boost::unit_test::test_suite* testSuite, boost::shared_ptr< ZipCodeDataFileRecordTestCases > tester ) const
    {
        AddTestCase( "ZipCodeDataFileRecord: TestCase_DFRConstructor", &TestCase_DFRConstructor<TPCE::ZipCodeDataFileRecord>, fields, testSuite );
        AddTestCaseField<TPCE::ZipCodeDataFileRecord, int>( "ZipCodeDataFileRecord: TestCase_DivisionTaxKey", &TestCase_DFRField<TPCE::ZipCodeDataFileRecord, int>, fields, &TPCE::ZipCodeDataFileRecord::DivisionTaxKey, atoi(divisionTaxKey.c_str()), testSuite );
        AddTestCaseField<TPCE::ZipCodeDataFileRecord, const std::string&>( "ZipCodeDataFileRecord: TestCase_ZC_CODE", &TestCase_DFRField<TPCE::ZipCodeDataFileRecord, const std::string&>, fields, &TPCE::ZipCodeDataFileRecord::ZC_CODE, zc_code, testSuite );
        AddTestCaseField<TPCE::ZipCodeDataFileRecord, const std::string&>( "ZipCodeDataFileRecord: TestCase_ZC_TOWN", &TestCase_DFRField<TPCE::ZipCodeDataFileRecord, const std::string&>, fields, &TPCE::ZipCodeDataFileRecord::ZC_TOWN, zc_town, testSuite );
        AddTestCaseField<TPCE::ZipCodeDataFileRecord, const std::string&>( "ZipCodeDataFileRecord: TestCase_ZC_DIV", &TestCase_DFRField<TPCE::ZipCodeDataFileRecord, const std::string&>, fields, &TPCE::ZipCodeDataFileRecord::ZC_DIV, zc_div, testSuite );
        AddTestCase( "ZipCodeDataFileRecord: TestCase_DFRToString", &TestCase_DFRMultiFieldToString<TPCE::ZipCodeDataFileRecord>, fields, testSuite );
    }

} // namespace EGenUtilitiesTest
