# Automated Financial Analysis Report

## Metadata
- chapter: ch15
- pipeline: multi-agent
- stock_code: 000001

## Executive Summary
analyst completed task with 3/3 successful tool calls.

## Review
- status: success
- reviewer_note: review passed
- warnings: 无

## Evidence Map
| id | source | claim | evidence |
|---|---|---|---|
| 1 | get_indicators | Output from get_indicators | {'stock_code': '000001', 'window': [2021, 2023], 'current_ratio': 1.92, 'debt_ratio': 0.47} |
| 2 | get_risk_score | Output from get_risk_score | {'stock_code': '000001', 'risk_score': 0.22, 'level': 'low'} |
| 3 | get_esg_score | Output from get_esg_score | {'stock_code': '000001', 'total_score': 0.74, 'grade': 'A-'} |

## Boundaries
- This report is auto-generated from tool outputs.
- High-risk decisions require human review.
