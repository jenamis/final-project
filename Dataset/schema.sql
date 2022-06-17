﻿-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE "Sample_outcomes" (
    "sample_id" varchar   NOT NULL,
    "structure_id" varchar   NOT NULL,
    "preferred_lcms_method" varchar   NOT NULL,
    "spe_method" varchar   NOT NULL,
    "method" varchar   NOT NULL,
    "spe_successful" varchar   NOT NULL,
    "crashed_out" varchar   NOT NULL,
    "sample_status" varchar   NOT NULL,
    "sample_current_status" varchar   NOT NULL,
    "termination_cause" varchar   NOT NULL,
    "termination_step" varchar   NOT NULL,
    "termination_details" varchar   NOT NULL,
    "reaction_scale_(mmol)" float   NOT NULL,
    "selected_fractions" varchar   NOT NULL,
    "volumn_collecteda_(ml)" float   NOT NULL,
    "total_fractions_collected" int   NOT NULL,
    "recovered_sample_dry_mass_(mg)" float   NOT NULL,
    "precent_yield" float   NOT NULL,
    "%_purity_(by_lcms)" float   NOT NULL,
    "purification_comments" varchar   NOT NULL,
    CONSTRAINT "pk_Sample_outcomes" PRIMARY KEY (
        "sample_id"
     )
);

CREATE TABLE "Structures" (
    "structure_id" varchar   NOT NULL,
    "MolWt" float   NOT NULL,
    "exactMolWt" float   NOT NULL,
    "logP" float   NOT NULL,
    "HBD" int   NOT NULL,
    "HBA" int   NOT NULL,
    "TPSA" float   NOT NULL,
    "Flexibility" float   NOT NULL,
    "Rotatable_bonds" int   NOT NULL,
    "qed" float   NOT NULL,
    "HeavyAtomMolWt" float   NOT NULL,
    "MolLogP" float   NOT NULL,
    "FractionCSP3" float   NOT NULL,
    "NumValenceElectrons" int   NOT NULL,
    "MaxPartialCharge" float   NOT NULL,
    "MinPartialCharge" float   NOT NULL,
    "FpDensityMorgan1" float   NOT NULL,
    "BalabanJ" float   NOT NULL,
    "BertzCT" float   NOT NULL,
    "HallKierAlpha" float   NOT NULL,
    "Ipc" float   NOT NULL,
    "Kappa2" float   NOT NULL,
    "LabuteASA" float   NOT NULL,
    "PEOE_VSA10" float   NOT NULL,
    "PEOE_VSA2" float   NOT NULL,
    "SMR_VSA10" float   NOT NULL,
    "SMR_VSA4" float   NOT NULL,
    "SlogP_VSA2" float   NOT NULL,
    "SlogP_VSA6" float   NOT NULL,
    "MaxEStateIndex" float   NOT NULL,
    "MinEStateIndex" float   NOT NULL,
    "EState_VSA3" float   NOT NULL,
    "EState_VSA8" float   NOT NULL,
    "HeavyAtomCount" int   NOT NULL,
    "NHOHCount" int   NOT NULL,
    "NOCount" int   NOT NULL,
    "NumAliphaticCarbocycles" int   NOT NULL,
    "NumAliphaticHeterocycles" int   NOT NULL,
    "NumAliphaticRings" int   NOT NULL,
    "NumAromaticCarbocycles" int   NOT NULL,
    "NumAromaticHeterocycles" int   NOT NULL,
    "NumAromaticRings" int   NOT NULL,
    "NumHAcceptors" int   NOT NULL,
    "NumHDonors" int   NOT NULL,
    "NumHeteroatoms" int   NOT NULL,
    "NumRotatableBonds" int   NOT NULL,
    "NumSaturatedCarbocycles" int   NOT NULL,
    "NumSaturatedHeterocycles" int   NOT NULL,
    "NumSaturatedRings" int   NOT NULL,
    "RingCount" int   NOT NULL,
    CONSTRAINT "pk_Structures" PRIMARY KEY (
        "structure_id"
     )
);

ALTER TABLE "Sample_outcomes" ADD CONSTRAINT "fk_Sample_outcomes_structure_id" FOREIGN KEY("structure_id")
REFERENCES "Structures" ("structure_id");

