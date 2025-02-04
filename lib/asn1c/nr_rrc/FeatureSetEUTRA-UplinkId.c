/*
 * Copyright 2020-2021 Telecom Paris
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
/*
 * Generated by asn1c-0.9.29 (http://lionet.info/asn1c)
 * From ASN.1 module "NR-RRC-Definitions"
 * 	`asn1c -gen-PER -fcompound-names -findirect-choice -no-gen-example`
 */

#include "FeatureSetEUTRA-UplinkId.h"

int
FeatureSetEUTRA_UplinkId_constraint(const asn_TYPE_descriptor_t *td, const void *sptr,
			asn_app_constraint_failed_f *ctfailcb, void *app_key) {
	long value;
	
	if(!sptr) {
		ASN__CTFAIL(app_key, td, sptr,
			"%s: value not given (%s:%d)",
			td->name, __FILE__, __LINE__);
		return -1;
	}
	
	value = *(const long *)sptr;
	
	if((value >= 0 && value <= 256)) {
		/* Constraint check succeeded */
		return 0;
	} else {
		ASN__CTFAIL(app_key, td, sptr,
			"%s: constraint failed (%s:%d)",
			td->name, __FILE__, __LINE__);
		return -1;
	}
}

/*
 * This type is implemented using NativeInteger,
 * so here we adjust the DEF accordingly.
 */
static asn_oer_constraints_t asn_OER_type_FeatureSetEUTRA_UplinkId_constr_1 CC_NOTUSED = {
	{ 2, 1 }	/* (0..256) */,
	-1};
asn_per_constraints_t asn_PER_type_FeatureSetEUTRA_UplinkId_constr_1 CC_NOTUSED = {
	{ APC_CONSTRAINED,	 9,  9,  0,  256 }	/* (0..256) */,
	{ APC_UNCONSTRAINED,	-1, -1,  0,  0 },
	0, 0	/* No PER value map */
};
static const ber_tlv_tag_t asn_DEF_FeatureSetEUTRA_UplinkId_tags_1[] = {
	(ASN_TAG_CLASS_UNIVERSAL | (2 << 2))
};
asn_TYPE_descriptor_t asn_DEF_FeatureSetEUTRA_UplinkId = {
	"FeatureSetEUTRA-UplinkId",
	"FeatureSetEUTRA-UplinkId",
	&asn_OP_NativeInteger,
	asn_DEF_FeatureSetEUTRA_UplinkId_tags_1,
	sizeof(asn_DEF_FeatureSetEUTRA_UplinkId_tags_1)
		/sizeof(asn_DEF_FeatureSetEUTRA_UplinkId_tags_1[0]), /* 1 */
	asn_DEF_FeatureSetEUTRA_UplinkId_tags_1,	/* Same as above */
	sizeof(asn_DEF_FeatureSetEUTRA_UplinkId_tags_1)
		/sizeof(asn_DEF_FeatureSetEUTRA_UplinkId_tags_1[0]), /* 1 */
	{ &asn_OER_type_FeatureSetEUTRA_UplinkId_constr_1, &asn_PER_type_FeatureSetEUTRA_UplinkId_constr_1, FeatureSetEUTRA_UplinkId_constraint },
	0, 0,	/* No members */
	0	/* No specifics */
};

