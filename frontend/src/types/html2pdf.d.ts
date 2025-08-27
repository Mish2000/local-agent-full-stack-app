// src/types/html2pdf.d.ts
declare module "html2pdf.js" {
    export interface Html2CanvasOptions {
        backgroundColor?: string | null;
        scale?: number;
        useCORS?: boolean;
        scrollX?: number;
        scrollY?: number;
        windowWidth?: number;
        windowHeight?: number;
    }

    export type JsPDFUnit = "pt" | "mm" | "cm" | "in";
    export type JsPDFOrientation = "portrait" | "landscape";

    export interface JsPDFOptions {
        unit?: JsPDFUnit;
        format?: string | [number, number];
        orientation?: JsPDFOrientation;
    }

    export interface Html2PdfImageOptions {
        type?: "jpeg" | "png" | "webp";
        quality?: number; // 0..1
    }

    export interface PagebreakOptions {
        mode?: Array<"css" | "legacy" | "avoid-all">;
    }

    export interface Html2PdfOptions {
        margin?: number | [number, number] | [number, number, number, number];
        filename?: string;
        image?: Html2PdfImageOptions;
        html2canvas?: Html2CanvasOptions;
        jsPDF?: JsPDFOptions;
        pagebreak?: PagebreakOptions;
    }

    export interface Html2PdfInstance {
        set(options: Html2PdfOptions): Html2PdfInstance;
        from(element: HTMLElement): Html2PdfInstance;
        save(): Promise<void>;
    }

    export interface Html2PdfFactory {
        (): Html2PdfInstance;
    }

    const html2pdf: Html2PdfFactory;
    export default html2pdf;
}
