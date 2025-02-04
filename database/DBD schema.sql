﻿-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE "outcomes" (
    "sample_id" varchar   NOT NULL,
    "structure_id" varchar   NOT NULL,
    "preferred_lcms_method" varchar,
    "spe_method" varchar,
    "method" varchar,
    "spe_successful" varchar,
    "crashed_out" varchar,
    "sample_status" varchar,
    "sample_current_status" varchar,
    "termination_cause" varchar,
    "termination_step" varchar,
    "termination_details" varchar,
    "reaction_scale_(mmol)" float,
    "selected_fractions" varchar,
    "volumn_collected_(ml)" float,
    "total_fractions_collected" int,
    "recovered_sample_dry_mass_(mg)" float,
    "precent_yield" float,
    "%_purity_(by_lcms)" float,
    "purification_comments" varchar,
    CONSTRAINT "pk_outcomes" PRIMARY KEY (
        "sample_id"
     )
);

CREATE TABLE "structures" (
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
    "MolMR" float   NOT NULL,
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

ALTER TABLE "outcomes" ADD CONSTRAINT "fk_outcomes_structure_id" FOREIGN KEY("structure_id")
REFERENCES "structures" ("structure_id");